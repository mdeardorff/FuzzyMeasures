"""Module handling the Scorer class."""

import copy
import json
import logging
import math
import multiprocessing
import os
import platform
import queue
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from inspect import signature
from typing import Dict, List, Any

import matplotlib as mpl
import numpy as np
import pandas as pd

import atlas_scorer.models as models
from atlas_scorer.errors import AtlasScorerError
from atlas_scorer.roc import atlas_score_roc
from atlas_scorer.schemas import AtlasDeclSchema, AtlasTruthSchema


@unique
class AggMetadataEnum(Enum):
    confidence = "confidence"
    labels = "labels"
    tp_annotations = "tp_annotations"
    fa_annotations = "fa_annotations"
    missed_annotations = "missed_annotations"
    tp_declarations = "tp_declarations"
    tp_declarations_agg = "tp_declarations_agg"
    fa_declarations = "fa_declarations"
    annotations_to_declarations = "annotations_to_declarations"
    declarations_to_annotations = "declarations_to_annotations"


def sort_dict_by_keys(d):
    keys = sorted(d.keys())
    ret_dict = OrderedDict()
    for k in keys:
        ret_dict[k] = d[k]
    return ret_dict


def link_worker(
    ann_tbl: pd.DataFrame,
    decl_tbl: pd.DataFrame,
    uid_q_in: multiprocessing.Queue,
    det_link_q: multiprocessing.Queue,
    id_link_q: multiprocessing.Queue,
):

    while True:
        # Grab uid to process
        uid = uid_q_in.get()

        # Poison-pill value of `None` results in terminating worker
        if uid is None:
            break

        cur_anno = ann_tbl[ann_tbl.uid.values == uid]
        cur_decl = decl_tbl[decl_tbl.uid.values == uid]

        anno_frames = cur_anno.frame.values
        decl_frames = cur_decl.frame.values

        # For each frame, frame_anno and frame_decl are relevant.  We don't
        # have to loop over decl_frames, since any decl with decl_frame not
        # present in anno_frames can't be linked to any annots
        for frame in np.unique(anno_frames):
            frame_anno = cur_anno[anno_frames == frame]
            frame_decl = cur_decl[decl_frames == frame]

            # Note that due to PANDAS index magic, the resulting
            # links and id_links map key values are the global indices, and
            # not simply [1 ... len(frame_anno)]
            links, id_links = Scorer._link_frame(frame_anno, frame_decl)
            det_link_q.put(links)
            id_link_q.put(id_links)


class Scorer:
    """Atlas Scorer class for linking and generating ROC curves.

    Args:
        frame_remove_function (func, optional): A function that takes a frame
            index (int) and uid (str) and returns either True or False.  Frames
            where frame_remove_function returns True are removed during
            Decl/Truth loading.  Note: to save memory this function is applied
            once during the `Scorer.load()` method and unlike other functions
            (below), changing this function requires re-loading the DECL and
            TRUTH files, then re-linking; Default function does not discard
            any frames

        annotation_remove_function (func, optional): If this returns `True`,
            do not consider the annotation during aggregation; Default function
            includes all annotations during aggregation

        declaration_remove_function (func, optional): If this returns `True`,
            do not consider the declaration during aggregation; Default function
            includes all declarations during aggregation

        annotation_binary_label_function (func, optional): Returns a binary
            label for an annotation. Setting to zero will then result in
            declarations linked to the annotation counting as false alarms
            during aggregation; Default function returns value of `True / 1`
            for all annotations.

        annotation_ignorable_function (func, optional): If return value is
            `True`, do not count this annotation as a possible detection (do
            not increment or penalize PD) and do not count any declarations
            that are ONLY linked yo ignorable annotations as false positives;
            Default function returns `False` for all annotations

        annotation_class_alias_dict (dict, optional): A dictionary with keys as
            original class labels in the annotation, and value as a string label
            to use in-place of original annotation class-label for the purposes
            of scoring. For example, one might use the following dictionary:
                {'PICKUP': 'CIVILIAN VEHICLE', 'SUV': 'CIVILIAN VEHICLE'}
            to score an algorithm trained to detect the `CIVILIAN VEHICLE` class

        declaration_class_alias_dict (dict, optional): A dictionary with keys as
            original class labels in the declaration, and value as a string
            label to use in-place of original declaration class-label for the
            purposes of scoring. For example, one might use the following
            dictionary:
                {'PICKUP_F350': 'PICKUP'}
            to score an algorithm trained to detect the `PICKUP_F350` class but
            evaluated on a collection where the matching class is labeled as
            'PICKUP'

        num_workers (int, optional): Number of multiprocessing workers to
            instantiate to speed up linking. Default of `None` results in
            `max(os.cpu_count() - 1, 1)` workers being instantiated
            in LINUX, and 1 worker in WINDOWS.
            Note that setting num_workers > 1 invokes multi-processing which
            requires the use of a __main__() method in your script.  Failure
            to use a __main__() method will cause errors in Windows (and
            possibly in LINUX depending on whether multi-processing is
            configured to use FORK or SPAWN).

    Properties:
        declarations_table: declarations in a pandas format keeping only
            scoring relevant data

        annotations_table: annotations in a pandas format keeping only
            scoring relevant data

        det_link_map: OrderedDict - a sparse mapping from annotations to linked
            declarations (See link() for more information)

        id_link_map: OrderedDict - a sparse mapping from annotations to linked
            declarations (See link() for more information)

        validate_csv - Settable property, validate incoming csv files if `True`

    Example Usage:
        # Example usage:
        # truthFiles and declFiles are lists of .truth.json and
        # .decl.json files
        s = Scorer()
        s.load(truth_files, decl_files)
        roc = s.score()
        plt.plot(roc.far, roc.pd) # with, e.g., matplotlib as plt

        # This is equivalent to:
        s = Scorer()
        s.load(truth_files, decl_files)
        s.link()
        roc, _ = s.aggregate()

        # Ignore alarms on distant targets, and only score every 30th
        # frame
        s = Scorer(annotation_ignorable_function=lambda a: a['range'] > 5500)
        s.frame_remove_function = lambda i: i % 30 > 0
        s.load(truth_files, decl_files)
        s.link()
        roc, _ = s.aggregate()
        plt.plot(roc.far, roc.pd) # with, e.g., matplotlib as plt

        # Less memory usage:
        s = Scorer(annotation_ignorable_function=lambda a: a['range'] > 5500)
        roc = s.roc_from_files(truth_files, decl_files)

    Note:
        The scoring functionality is modularized into two steps:
            `link()`
            `aggregate()`
        See the corresponding doc-strings for these methods for details

        If it is desirable to change how annotations are linked to declarations,
        Scorer should be subclassed, and `_link_frame` should be overridden.
        This Scorer should be thought of as the 'default' scorer, while scorers
        with other linking methods can be used with minimal code changes.

        In addition, several function handles are used to determine how to
        handle each annotation and declaration during scoring. The rows of
        the pandas DataFrame will be passed to each function and can
        be used to make specific changes to the scoring based on metadata
            e.g. ignoring annotations >5km.
    """

    #: Logger
    LOG = logging.getLogger(__name__)
    DEFAULT_WORKERS = (
        1 if platform.system() == "Windows" else max(os.cpu_count() - 1, 1)
    )

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        if value is None:
            self._num_workers = self.DEFAULT_WORKERS
        else:
            self._num_workers = max(1, value)

    @property
    def frame_remove_function(self):
        """
            A frame_remove_function is a lambda function of the form:
                bool = fn(frame_num)
            or
                bool = fn(frame_num, uid)

            where frame_num is the 1-based index of the frame, and uid is a
            string representing the uid.  Internally, frame_remove_function
            is always stored as a 2-parameter function, but the second argument
            (uid) is ignored if the provided function had only 1 parameter
            """
        return self._frame_remove_function

    @frame_remove_function.setter
    def frame_remove_function(self, fn):
        """
            Introspect frame_remove_function to check if it is old-style with
            only single argument (frame_num) and wrap it to make it compatible
            with the new (frame_num, uid) call signature; the resulting
            frame_remove_function property is always a 2-parameter function of
            the form:
              fn(frame_index, uid)
            """
        sig = signature(fn)
        if len(sig.parameters) == 1:
            self._frame_remove_function = lambda i, uid: fn(i)
        else:
            self._frame_remove_function = fn

    def __init__(
        self,
        frame_remove_function=lambda i, uid: False,
        annotation_remove_function=lambda a: False,
        declaration_remove_function=lambda d: False,
        annotation_binary_label_function=lambda a: True,
        annotation_ignorable_function=lambda a: False,
        annotation_class_alias_dict=None,
        declaration_class_alias_dict=None,
        validate_csv=True,
        num_workers=None,
    ):

        self.validate_csv = validate_csv
        self.frame_remove_function = frame_remove_function

        self.annotation_remove_function = annotation_remove_function
        self.declaration_remove_function = declaration_remove_function
        self.annotation_binary_label_function = annotation_binary_label_function
        self.annotation_ignorable_function = annotation_ignorable_function
        self.annotation_class_alias_dict = (
            annotation_class_alias_dict
            if annotation_class_alias_dict is not None
            else dict()
        )
        self.declaration_class_alias_dict = (
            declaration_class_alias_dict
            if declaration_class_alias_dict is not None
            else dict()
        )

        self.num_workers = num_workers

        self.det_link_map = OrderedDict()
        self.id_link_map = OrderedDict()
        self.declarations_table = None
        self.annotations_table = None
        self.num_frames = 0
        self.num_frames_by_uid = {}
        self.num_frames_by_uid_pre_remove = {}

    def reset(self):
        """Return scoring variables to initial state."""
        self.det_link_map = OrderedDict()
        self.id_link_map = OrderedDict()
        self.declarations_table = None
        self.annotations_table = None
        self.num_frames = 0
        self.num_frames_by_uid = {}
        self.num_frames_by_uid_pre_remove = {}

    def declaration_objects_to_table(self, decl_objs):
        """
        Convert declaration objects to a pandas table.

        Args:
            decl_objs (List of dicts or AtlasDecl): List of dicts or
                AtlasDecl objects

        Returns:
            (pd.DataFrame): Pandas DataFrame containing contents of
            ``decl_objs`` including uid, frame, bbox, confidence, and class
        """

        # Ensure we have a list of declaration and truth objects.
        if not isinstance(decl_objs, list):
            decl_objs = [decl_objs]

        # During loading, if validate_inputs=True, the DECL and ANNOT files are
        # read using our SCHEMA to ensure JSON validity.
        # These generate a list of decl_objs.  If validate_inputs=False,
        # we instead get a dictionary. Not validating is MUCH faster for very
        # large JSON files, so we handle both cases with the is_dict flag.
        is_dict = False
        if isinstance(decl_objs[0], dict):
            is_dict = True

        uids = []
        frames = []
        bboxes = []
        confidences = []
        classes = []
        user_datas = []

        # Throughout, use:
        #   val = o['str'] if is_dict else o.str
        # to handle both input data types
        for i_decl, obj in enumerate(decl_objs):
            uid = obj["fileUID"] if is_dict else obj.uid
            frame_decls = (
                obj["frameDeclarations"] if is_dict else obj.frameDeclarations
            )

            for frame_str, frame_decl in frame_decls.items():
                frame_num = int(frame_str[1:])

                decl_vec = (
                    frame_decl["declarations"]
                    if is_dict
                    else frame_decl.declarations
                )
                num_decl = len(decl_vec)
                uids.extend(num_decl * [uid])
                frames.extend(num_decl * [frame_num])

                if not is_dict:
                    for d in decl_vec:
                        classes.append(
                            self.declaration_class_alias_dict.get(
                                d.obj_class, d.obj_class
                            )
                        )
                        bboxes.append(d.shape)
                        confidences.append(d.confidence)
                        user_datas.append(d.userData)
                else:
                    for d in decl_vec:
                        classes.append(
                            self.declaration_class_alias_dict.get(
                                d["class"], d["class"]
                            )
                        )
                        bboxes.append(self._shape_to_object(d["shape"]))
                        confidences.append(d["confidence"])
                        user_datas.append(d.get("userData", {}))

        return pd.DataFrame(
            {
                "uid": uids,
                "frame": frames,
                "bbox": bboxes,
                "confidence": confidences,
                "class": classes,
                "userData": user_datas,
            }
        )

    def annotation_objects_to_table(self, truth_objs):
        """
        Convert annotation objects to a pandas table.

        Args:
            truth_objs (List of dict or AtlasTruth): List of dicts or
                AtlasTruth objects

        Returns:
            (pd.DataFrame): Pandas DataFrame containing contents of
            ``truth_objs`` including uid, frame, bbox, range, aspect, and class
        """

        # Ensure we have list of declaration and truth objects.
        if not isinstance(truth_objs, list):
            truth_objs = [truth_objs]

        # During loading, if validate_inputs=True, the DECL and ANNOT files are
        # read using our SCHEMA to ensure JSON validity.  These generate a list
        # of decl_objs.  If validate_inputs=False, we instead get a dictionary.
        # Not validating is MUCH faster for very large JSON files, so we handle
        # both cases with the is_dict flag.
        is_dict = False
        if isinstance(truth_objs[0], dict):
            is_dict = True

        uids = []
        frames = []
        bboxes = []
        ranges = []
        aspects = []
        classes = []
        user_datas = []

        # Throughout, use:
        #   val = o['str'] if is_dict else o.str
        # to handle both input data types
        for obj in truth_objs:
            uid = obj["fileUID"] if is_dict else obj.uid
            frame_annots = (
                obj["frameAnnotations"] if is_dict else obj.frameAnnotations
            )

            for frame_str, frame_anno in frame_annots.items():
                frame_num = int(frame_str[1:])

                anno_vec = (
                    frame_anno["annotations"]
                    if is_dict
                    else frame_anno.annotations
                )
                num_anno = len(anno_vec)

                if num_anno > 0:
                    uids.extend(num_anno * [uid])
                    frames.extend(num_anno * [frame_num])

                    if not is_dict:
                        for a in anno_vec:
                            classes.append(
                                self.annotation_class_alias_dict.get(
                                    a.obj_class, a.obj_class
                                )
                            )
                            bboxes.append(a.shape)
                            ranges.append(a.range)
                            aspects.append(a.aspect)
                            user_datas.append(a.userData)
                    else:
                        for a in anno_vec:
                            classes.append(
                                self.annotation_class_alias_dict.get(
                                    a["class"], a["class"]
                                )
                            )
                            bboxes.append(self._shape_to_object(a["shape"]))
                            # Some annotations may not have a range.  For
                            # example, for DEER and other animals the range to
                            # the object is not known.  If the 'range' field is
                            # not present, we replace with None.  The same is
                            # true for aspect.
                            ranges.append(a.get("range", None))
                            aspects.append(a.get("aspect", None))
                            user_datas.append(a.get("userData", {}))

        return pd.DataFrame(
            {
                "uid": uids,
                "frame": frames,
                "bbox": bboxes,
                "range": ranges,
                "aspect": aspects,
                "class": classes,
                "userData": user_datas,
            }
        )

    def _check_input_files(self, truth_files, decl_files):
        """Check that input files meet basic requirements."""

        # Ensure we have list group of declaration and truth file names.
        if not isinstance(decl_files, list):
            decl_files = [decl_files]

        if not isinstance(truth_files, list):
            truth_files = [truth_files]

        if len(decl_files) != len(truth_files):
            raise AtlasScorerError(
                "Must have same number of truth and declaration files."
            )

        if not decl_files:
            raise AtlasScorerError("Must have at least one declaration file.")

    def load(self, truth_files, decl_files, validate_inputs=False):
        """
        Populate scorer object with data from the lists truth_files and
        decl_files.  truth_files and decl_files must be the same length, and
        in the same order.

        Args:
            truth_files (List): File names of truth files to load. The
                order of the truth_files must match the order of the decl_files
                (i.e., the UID of truth_file[i] must match the UID of
                decl_file[i])
            decl_files (List): File names of declarations files to load. The
                order of the decl_files must match the order of the truth_files
                (i.e., the UID of truth_file[i] must match the UID of
                decl_file[i])
            validate_inputs (bool, optional): Whether to validate truth and decl
                files against Schema. Setting this to ``True`` will make loading
                substantially slower; Default **False**

        Returns:
            self with updated num_frames, declarations_table and
            annotations_table.
        """
        self._check_input_files(truth_files, decl_files)
        self.reset()

        return_dict = not validate_inputs
        decl_tables = []
        anno_tables = []
        for (truth_file, decl_file) in zip(truth_files, decl_files):
            truth_obj = self.load_truth_file(
                truth_file, return_dict=return_dict
            )
            decl_obj = self.load_decl_file(decl_file, return_dict=return_dict)

            uid = truth_obj["fileUID"] if return_dict else truth_obj.uid
            uid_d = decl_obj["fileUID"] if return_dict else decl_obj.uid

            # We raise this error because the frame_remove_function below
            # assumes the UIDs are in the same order; this is required to
            # ensure that frame_remove_function gets applied to the right
            # frames from the right DECLs and TRUTHs (annotations)
            if uid != uid_d:
                raise ValueError(
                    f"The input DECL and TRUTH files to load() "
                    f"must have UIDs in the same order, but files "
                    f"{truth_file} and {decl_file} contained uids "
                    f"{uid} and {uid_d}"
                )

            cur_decl = self.declaration_objects_to_table([decl_obj])
            cur_anno = self.annotation_objects_to_table([truth_obj])

            n_frames = (
                truth_obj["nFrames"] if return_dict else truth_obj.nFrames
            )
            n_frames = int(n_frames)

            if np.any(cur_anno["frame"] > n_frames):
                raise ValueError(
                    f"The input {truth_file} had frame numbers "
                    f"{np.max(cur_anno['frame'])} which are "
                    f" greater than the specified {n_frames}"
                )

            if np.any(cur_decl["frame"] > n_frames):
                raise ValueError(
                    f"The input {decl_file} had frame numbers "
                    f"({np.max(cur_decl['frame'])}) which are "
                    f" greater than the {truth_file} specified {n_frames}"
                )

            remove = np.array(
                [
                    self.frame_remove_function(i + 1, uid)
                    for i in range(n_frames)
                ]
            )
            remove_count = remove.sum()
            self.num_frames_by_uid_pre_remove[uid] = n_frames
            self.num_frames_by_uid[uid] = n_frames - remove_count
            self.num_frames += self.num_frames_by_uid[uid]

            keep = np.logical_not(remove)
            cur_anno = cur_anno[keep[cur_anno.frame - 1]]
            cur_decl = cur_decl[keep[cur_decl.frame - 1]]

            decl_tables.append(cur_decl)
            anno_tables.append(cur_anno)

        # note that the index needs to be reset so that the linkMap properly
        # indexes into these tables.
        self.annotations_table = pd.concat(anno_tables, axis=0)
        self.annotations_table.reset_index(inplace=True, drop=True)

        self.declarations_table = pd.concat(decl_tables, axis=0)
        self.declarations_table.reset_index(inplace=True, drop=True)

        # Enforce integer dtype for 'frame' column
        self.annotations_table = self.annotations_table.astype({"frame": int})
        self.declarations_table = self.declarations_table.astype({"frame": int})

        return self

    def roc_from_files(self, truth_files, decl_files, validate_inputs=False):
        """
        Instantiate an AtlasMetricROC object from a set of truth and declaration
        files.

        :param List truth_files: File names of truth files to load.
        :param List decl_files: File names of declarations files to load.
        :param Bool validate_inputs: Whether to validate the JSON files (False)
        :return: An AtlasMetricROC object.
        :rtype: AtlasMetricROC
        """

        self._check_input_files(truth_files, decl_files)

        num_frames_total = 0
        conf_labels_total = pd.DataFrame()
        for i in range(len(decl_files)):
            iter_start = time.time()
            self.load(
                [truth_files[i]],
                [decl_files[i]],
                validate_inputs=validate_inputs,
            )
            self.link()
            _, agg_dict = self.aggregate()
            conf_labels_local = agg_dict["conf_labels"]
            conf_labels_total = pd.concat(
                (conf_labels_total, conf_labels_local)
            )
            num_frames_total = num_frames_total + self.num_frames

            self.LOG.info("Iteration %d took %f", i, time.time() - iter_start)

        roc = atlas_score_roc(
            conf_labels_total.confidence, conf_labels_total.label
        )
        roc.farDenominator = num_frames_total
        return roc

    def load_truth_file(self, truth_file, return_dict=True):
        """
        Convert truth_file into truth_obj.
        Args:
            truth_file (str): JSON truth file to load
            return_dict (bool, optional): Whether to return a dict data
                structure or an AtlasScorer object for the parsed truth
                file contents; Default is **True**

        Returns:
            (AtlasDecl or dict): Parsed contents of truth file
        """

        truth_schema = AtlasTruthSchema()
        try:
            with open(truth_file, "r") as fid:
                truth_json = json.load(fid)
        except FileNotFoundError as err:
            raise AtlasScorerError(f"Error loading {truth_file} -- {err}.")
        if return_dict:
            return truth_json
        return truth_schema.load(truth_json)

    def load_decl_file(self, decl_file, return_dict=True):
        """
        Convert decl_file into decl_obj.
        Args:
            decl_file (str): JSON or CSV declarations file to load
            return_dict (bool, optional): Whether to return a dict
                data structure or an AtlasScorer object for the parsed
                decl file contents; Default is **True**

        Returns:
            (AtlasDecl or dict): Parsed contents of decl file
        """

        decl_schema = AtlasDeclSchema()
        decl_file = str(decl_file)
        if decl_file.endswith(".csv"):
            decl_obj = decl_schema.load_csv(
                decl_file, validate=self.validate_csv, return_dict=return_dict
            )
        else:
            with open(decl_file, "r") as fid:
                decl_json = json.load(fid)
            if return_dict:
                return decl_json
            decl_obj = decl_schema.load(decl_json)
        return decl_obj

    def score(self):
        """Generate ROC object from current data, and set the linkMatrix in
        the scorer object.
        NOTE: this may throw out rows of the annotations_table.

        :return: An AtlasMetricROC object.
        :rtype: AtlasMetricROC
        """

        self.link()
        roc, _ = self.aggregate()

        return roc

    def link(self):
        """Determine linking structure for all annotations and declarations

        self.link() Generates self.det_link_map and self.id_link_map by
        applying _link_frame to all annotations and declarations that share
        the same uid and frame.

        The resulting self.det_link_map and self.id_link_map are OrderedDict
        objects where the keys correspond to the (Pandas) index of annotations
        in self.annotations_table and the entries are lists of (Pandas) indices
        into self.declarations_table, so map[i] contains j implies that
        declaration j was linked to annotation i.

        For example, if
            self.det_link_map[1789] == Int64Index([], dtype='int64'))
        then annotation index 1789 was linked to no declarations. Similarly
            self.det_link_map[1799] == Int64Index([243], dtype='int64')
        implies that annotation 1799 was linked to declaration 243, and
            self.det_link_map[1650] == Int64Index([101, 102], dtype='int64')
        implies that annotation 1650 was linked to declarations 101 and 102.

        Algorithm:
            For each combination of uid and frame index in all annotations*:
                c_annos = relevant annotations from self.annotations_table
                c_decls = relevant declarations from self.declarations_table
                links, id_links = self._link_frame(c_annos, c_decls)
                self.det_link_map.update(links)
                self.id_link_map.update(id_links)

        det_link_map vs. id_link_map:
            det_link_map will contain results of "detection" linking, e.g.,
            geometry-only based linking, irrespective of annotation and
            delaration "class" descriptors.

            id_link_map will contain results of "identification" linking, e.g.,
            id_link_map[i] contains j only if declaration j is geometrically
            linked to annotation i, AND declaration j and annotation i share
            the same class descriptor.

        *Note it is sufficient to iterate only across annotations unique uids
        and frame indices, as any declarations that are not from these
        uids/frame indices cannot possibly be included in the det_link_map nor
        id_link_map

        :return: Nothing
        """

        # End-users can easily mess with the declarations and annotations
        # tables, and this can be a really powerful tool for expert uses,
        # but doing so haphazardly will result in the index getting
        # out-of-whack.  This is bad, because in several places during and
        # after linking we assume the index in the tables is 0 to N.
        self.declarations_table.reset_index(inplace=True, drop=True)
        self.annotations_table.reset_index(inplace=True, drop=True)
        if self.num_workers == 1:
            self._link_single_worker()
        else:
            self._link_multi_worker()

    def _link_single_worker(self):
        """Single-process implementation of linking (reference implementation).
        This approach is much slower than `Scorer._link_multi_worker` but it
        does not rely on any multi-processing and can be relied on as a fallback
        in case of any issues with multi-processing on a system.
        """
        self.det_link_map = OrderedDict()
        self.id_link_map = OrderedDict()

        # Single-process implementation
        # For each UID, cur_anno and cur_decl are relevant
        for uid in np.unique(self.annotations_table.uid.values):
            cur_anno = self.annotations_table[
                self.annotations_table.uid.values == uid
            ]
            cur_decl = self.declarations_table[
                self.declarations_table.uid.values == uid
            ]

            anno_frames = cur_anno.frame.values
            decl_frames = cur_decl.frame.values

            # For each frame, frame_anno and frame_decl are relevant.  We don't
            # have to loop over decl_frames, since any decl with decl_frame not
            # present in anno_frames can't be linked to any annots
            for frame in np.unique(anno_frames):
                frame_anno = cur_anno[anno_frames == frame]
                frame_decl = cur_decl[decl_frames == frame]

                # Note that due to PANDAS index magic, the resulting
                # links and id_links map key values are the global indices, and
                # not simply [1 ... len(frame_anno)]
                links, id_links = self._link_frame(frame_anno, frame_decl)

                self.det_link_map.update(links)
                self.id_link_map.update(id_links)

    def _link_multi_worker(self):
        """Multiprocessing based implementation of linking that results in a
        huge boost in linking speeds compared to the single-process
        implementation in `Scorer._link_single_worker`. Results have been
        verified to match those of `Scorer._link_single_worker`"""
        self.det_link_map = OrderedDict()
        self.id_link_map = OrderedDict()

        uid_q_in = multiprocessing.Queue()
        det_link_q = multiprocessing.Queue()
        id_link_q = multiprocessing.Queue()

        # Instantiate and start MP Consumer processes
        processes = []
        for i in range(self.num_workers):
            proc = multiprocessing.Process(
                target=link_worker,
                daemon=False,
                args=(
                    self.annotations_table,
                    self.declarations_table,
                    uid_q_in,
                    det_link_q,
                    id_link_q,
                ),
            )
            processes.append(proc)
            proc.start()

        # Initialize uid queue to start processing
        for uid in np.unique(self.annotations_table.uid.values):
            uid_q_in.put(uid)
        # Add poison-pill values at end to terminate all workers/Consumer procs
        for i in range(self.num_workers):
            uid_q_in.put(None)

        # Pull out all links and id_links from the queues
        # Note: The link processes above continuously add entries into
        # det_link_q and id_link_q and will not terminate till the items in
        # those queues have been consumed. The loop below ensures that items are
        # continuously pulled off these queues and stored in a list. This
        # solves two problems:
        #     1. It prevents the output queues getting too full and blocking if
        #       a large amount of data is being linked and no active process is
        #       consuming entries from these queues
        #     2. It enforces that the processes above can terminate by
        #       themselves once the queues they are feeding get fully consumed
        links_list, id_links_list = [], []
        while True:
            try:
                links_list.append(det_link_q.get_nowait())
            except queue.Empty:
                empty1 = True
            else:
                empty1 = False

            try:
                id_links_list.append(id_link_q.get_nowait())
            except queue.Empty:
                empty2 = True
            else:
                empty2 = False

            # Only exit this loop if both queues are empty and
            # all processes feeding them have terminated since this
            # means that we successfully consumed all the data that
            # was put on the queues by the processes.
            if empty1 and empty2:
                if any([p.is_alive() for p in processes]):
                    time.sleep(0.1)
                else:
                    break

        assert len(links_list) == len(id_links_list), (
            "Multiprocessing linking resulted in different number of links "
            "and id_links dict objects!"
        )

        # Update link maps
        for det_links in links_list:
            self.det_link_map.update(det_links)
        for id_links in id_links_list:
            self.id_link_map.update(id_links)

        # Sort link map dicts
        self.det_link_map = sort_dict_by_keys(self.det_link_map)
        self.id_link_map = sort_dict_by_keys(self.id_link_map)

    @staticmethod
    def _link_frame(annotations, declarations):
        """Link the annotations and declarations within a specific frame

        self._link_frame(self, annotations, declarations) returns two
        OrderedDict objects: det_links, id_links.

        Each return value is a mapping from annotation indices to (possibly
        empty) lists of declaration indices.

        If map[i] contains j, this indicates that declaration j was linked to
        annotation i.  If map[i] = [j,k,l], all of declarations j,k,l were
        linked to annotation i.  If map[i] is empty, no declarations were mapped
        to annotation i.

        det_links will contain results of "detection" linking (e.g., an
        annotation and declaration are linked iff their geometric
        representations overlap sufficiently).

        id_links will contain results of "identification" linking (e.g., an
        annotation and declaration are linked if they are "Detection linked" AND
        their class-strings match).

        Note: Various alternative linking logic strategies can be implemented
        by overloading this method to, e.g., use IOU bounding box logic, or
        to implement more advanced class-based linking.

        TODO: Separate out new methods that explicitly allow geometric linking
            and identification-based linking so this whole method doesn't have
            to be re-written for each different linking method.

        :param annotations: An nAnnotations tall pandas DataFrame of annotations
        from the current frame. Note: the index of the
        annotations DataFrame must correspond to the global index from
        self.annotations_table
        :param declarations: An nDeclarations tall pandas DataFrame of
        declarations from the current frame.  Note: the index of the
        declarations DataFrame must correspond to the global index from
        self.declarations_table
        :returns:
            - (det_links, id_links) - tuple where det_links is an OrderedDict
            mapping of detections between annots and decls and id_links an
            OrderedDict mapping of identifications between annots and decls
        """

        PAD = 5

        # Compute centroid for each declaration.
        decl_center = declarations.bbox.apply(lambda x: x.center())
        decl_class = declarations["class"]

        det_links = OrderedDict()
        id_links = OrderedDict()
        for idx, anno in annotations.iterrows():
            # Generated padded bbox.
            bbox = anno.bbox
            pad_bbox = models.BBOX(
                [bbox.x - PAD, bbox.y - PAD, bbox.w + 2 * PAD, bbox.h + 2 * PAD]
            )

            # Determine which declarations are within the padded annotation
            # bbox.
            inside = decl_center.apply(pad_bbox.inside)
            det_links[idx] = inside[inside].index
            # id_links is class_match AND inside
            class_match = decl_class.apply(lambda x: x == anno["class"])
            class_match = np.logical_and(class_match, inside)
            id_links[idx] = class_match[class_match].index

        return det_links, id_links

    def aggregate(self, alg="detection", score_class=None):
        """Perform aggregation to convert annotations, declarations, and
        link_map entries into scored results.

        Args:
            score_class (str): A string corresponding to an annotation
                type (e.g., 'T-72') to score against for detection or
                identification.
            alg (str, optional): One of {'detection', 'identification'}; Default
                is `'detection'`

        Returns:
            tuple(roc, agg_metadata) - 2-element tuple where roc is an
                AtlasMetricROC and agg_metadata is a dict containing the
                following keys:
                'confidence': Array of confidences suitable for creating an
                    ROC curve.  The length of this list is dependent on the
                    number of annotations, declarations, and the linking matrix
                'labels': Array of labels suitable for creating an ROC curve.
                    The length of this list is dependent on the number of
                    annotations, declarations, and the linking matrix
                'tp_annotations': Array of len(annotations) indicating the
                    true-positive annotations for the purposes of aggregation
                'missed_annotations': Array of len(annotations) indicating the
                    TP annotations that should have been detected but were not
                    i.e., no declarations were linked to these TP annotations.
                'fa_annotations': Array of len(annotations) indicating the
                    false-alarm annotations for the purposes of aggregation
                'tp_declarations': Array of len(declarations) indicating which
                    declarations were linked to 1-or-more TP annotations. Note:
                    This includes TP declarations which were subsequently
                    discarded and not included in scoring if a higher conf
                    declaration was also linked to the same TP annotation.
                'tp_declarations_agg': Array of len(declarations) indicating
                    which declarations were linked to 1-or-more TP annotations
                    as the highest confidence declaration, i.e., every
                    declaration with a value of 1 in this array was responsible
                    for a TP detection in the aggregated ROC curve.
                'fa_declarations': Array of len(declarations) indicating which
                    declarations contributed as false-alarms (i.e., were not
                    linked to any annotations in tp_annotations)
                'annotations_to_declarations': Array of len(annotations)
                    where annotations_to_declarations[i] contains the index
                    of the highest-confidence declaration that was linked to the
                    i-th annotation, and -1 if no declaration was linked to the
                    i-th annotation
                'declarations_to_annotations': Array of len(declarations)
                    where declarations_to_annotations[i] contains the index
                    of the last annotation that was linked to the i-th
                    declaration, and -1 if no annotation was linked to the
                    i-th declaration

        scorer.aggregate() Performs basic "detection" type scoring and provides
        output conf_labels pandas DataFrame containing n confidence and binary
        label outputs in addition to an AltasMetricRoc object.  The relevant
        annotation-to-declaration linking map is based on .det_link_map.

        scorer.aggregate(alg='identification') Performs basic "identification"
        type scoring. The relevant annotation-to-declaration linking map is
        based on .id_link_map.

        scorer.aggregate(..., score_class=None) is the same as above.
        scorer.aggregate(..., score_class='T-72') generates ROC curves and
        scoring metrics for only the specified annotation class.

        Several user-settable atlasScorer functions can affect the aggregation
        process:

            declaration_remove_function: Return True if the declaration should
                be removed from consideration during scoring.
            annotation_remove_function: Return True if the annotation should be
                removed from consideration during scoring.
            annotation_binary_label_function: Return 1 if the annotation should
                be considered a "true detection" during scoring.
            annotation_ignorable_function: Return 1 if declarations on the
                annotation should be considered neither false alarms nor
                true-positives.

        During normal scoring, annotations where a linked declaration should
        count as a true-positive are determined by evaluating:
            label & ~ignorable & ~removed
        for each annotation.

        Annotations where a linked declaration should count as a false-positive
        are determined by evaluating:
            (~label & ~ignorable) | removed
        for each annotation.

        Declarations that should be counted as false alarms are then those
        declarations that are linked to zero annotations, or that are linked
        only to false-positive annotations.  Declarations that should be counted
        as detections are those declarations with the MAX confidence that are
        linked to a true-positive annotation.

        Note: if a score_class is specified, the following changes to the
        ignorable, label, and remove_decl values are made:

            Detection:
                # ignore non-score_class annotations that are "true positives"
                ignorable <- ignorable | (~a_is_class & label)
                # Score against the specified class
                label <- a_is_class

            Identification:
                # Remove irrelevant declarations
                remove_decl <- remove_decl | ~d_is_class
                # Score against the specified class
                label <- a_is_class

        TODO: Also output the indices of the declarations and/or annotations for
            each confidence/label in conf_labels
        """
        remove_decl = self.declarations_table.apply(
            self.declaration_remove_function, axis=1, result_type="reduce"
        )

        remove_annot = self.annotations_table.apply(
            self.annotation_remove_function, axis=1, result_type="reduce"
        )

        confidence = self.declarations_table.confidence.copy()
        labels = self.annotations_table.apply(
            self.annotation_binary_label_function, axis=1, result_type="reduce"
        )
        ignorable = self.annotations_table.apply(
            self.annotation_ignorable_function, axis=1, result_type="reduce"
        )

        confidence = confidence.to_numpy()
        labels = labels.to_numpy()
        ignorable = ignorable.to_numpy()

        if alg == "detection":
            link_matrix = self.det_link_map
            if score_class is not None:
                a_is_class = self.annotations_table["class"] == score_class
                ignorable = np.logical_or(
                    ignorable,
                    np.logical_and(np.logical_not(a_is_class), labels),
                )
                labels = a_is_class

        elif alg == "identification":
            link_matrix = self.id_link_map
            if score_class is not None:
                a_is_class = self.annotations_table["class"] == score_class
                d_is_class = self.declarations_table["class"] == score_class
                remove_decl = np.logical_or(
                    remove_decl, np.logical_not(d_is_class)
                )
                labels = a_is_class
        else:
            raise AtlasScorerError(
                f"Invalid alg provided to aggregation: {alg}"
            )

        tp_annotations = np.logical_and.reduce(
            (labels, np.logical_not(ignorable), np.logical_not(remove_annot))
        )
        fa_annotations = np.logical_or(
            np.logical_and(np.logical_not(labels), np.logical_not(ignorable)),
            remove_annot,
        ).to_numpy()

        # Occasionally, depending on how the .decl.json or .decl.csv files are
        # constructed, confidence can get created as an INT.
        # INTs are no good, since there is no np.nan equivalent.  Convert to
        # float
        confidence = confidence.astype(np.float)
        confidence[remove_decl] = np.nan

        # num declarations long vector storing whether a declaration was not
        # linked to an eligible annotation (not a fa_annotation)
        nolink_indicator = np.ones(confidence.shape)

        # Create TP declarations length indicator and confidence vector
        tp_confidence = np.full(self.annotations_table.shape[0], np.nan)
        tp_declarations_agg = np.zeros_like(confidence, dtype=bool)
        tp_declarations = np.zeros_like(confidence, dtype=bool)
        declarations_to_annotations = np.full_like(confidence, -1, dtype=int)
        missed_annotations = np.zeros_like(tp_annotations, dtype=bool)
        annotations_to_declarations = np.full_like(
            tp_annotations, -1, dtype=int
        )

        for annot_ind, decl_inds in link_matrix.items():
            if decl_inds.empty:  # No decl linked to this annotation
                # Note, tp_confidence is already nan

                # Flag as a missed TP if this annotation is a TP annotation
                if tp_annotations[annot_ind]:
                    missed_annotations[annot_ind] = 1
            else:
                # Flag all decl inds as linked (set 0) if this annotation is
                # not a false-alarm notation
                if not fa_annotations[annot_ind]:
                    # Set 0 via vectorized index notation
                    nolink_indicator[decl_inds] = 0

                # Find max conf declaration for linking to this annotation
                conf_argmax = confidence[decl_inds].argmax()
                annotations_to_declarations[annot_ind] = decl_inds[conf_argmax]
                tp_confidence[annot_ind] = confidence[decl_inds][conf_argmax]

                # Note that declarations that are on the annotation, regardless
                # of whether the annotation is a TP or not; note that since
                # multiple declarations can be on one annotation, this ends up
                # recording the index of the last annotation (in the
                # annotations_table) that each declaration was on
                for decl_ind in decl_inds:
                    declarations_to_annotations[decl_ind] = annot_ind

                # Update tp_declarations/tp_declaration_agg only if this
                # annotation is a TP annotation
                if tp_annotations[annot_ind]:
                    tp_declarations_agg[decl_inds[conf_argmax]] = 1
                    for decl_ind in decl_inds:
                        tp_declarations[decl_ind] = 1

        tp_confidence = np.array(tp_confidence)
        tp_confidence = tp_confidence[tp_annotations]

        fa_declarations = np.logical_and(
            nolink_indicator, np.logical_not(remove_decl)
        ).to_numpy()
        fa_confidence = confidence[fa_declarations]

        confidence = np.hstack((fa_confidence, tp_confidence))
        labels = np.hstack(
            (np.zeros_like(fa_confidence), np.ones_like(tp_confidence))
        )

        roc = atlas_score_roc(confidence, labels)
        roc.farDenominator = self.num_frames

        # Create dict of all relevant metadata of interest
        agg_metadata = {
            AggMetadataEnum.confidence.value: confidence,
            AggMetadataEnum.labels.value: labels,
            AggMetadataEnum.tp_annotations.value: tp_annotations,
            AggMetadataEnum.fa_annotations.value: fa_annotations,
            AggMetadataEnum.missed_annotations.value: missed_annotations,
            AggMetadataEnum.tp_declarations.value: tp_declarations,
            AggMetadataEnum.tp_declarations_agg.value: tp_declarations_agg,
            AggMetadataEnum.fa_declarations.value: fa_declarations,
            AggMetadataEnum.annotations_to_declarations.value:
                annotations_to_declarations,
            AggMetadataEnum.declarations_to_annotations.value:
                declarations_to_annotations,
        }
        return roc, agg_metadata

    @staticmethod
    def _shape_to_object(shape):
        """
        Convert shape dict into the appropriate Shape object instance.

        Args:
            shape (dict): Of the form: { "type": "bbox_xywh", "data": [...] }

        Returns:
            (Shape instance): The shape instance
        """
        shape_to_class_map = {
            "bbox_xywh": models.BBOX,
        }
        shape_type = shape.get("type", None)
        if shape_type is None:
            raise AtlasScorerError(f"No shape type provided: {shape}")
        if shape_type not in shape_to_class_map:
            raise AtlasScorerError(
                f"Unsupported shape type: {shape_type} provided"
            )
        return shape_to_class_map[shape_type](shape.get("data", [np.nan] * 4))

    def copy(self):
        """Return deep-copy of this `Scorer` instance"""
        return copy.deepcopy(self)


@dataclass
class ConfusionMatrixResult:
    normalize: bool
    threshold: float
    count_matrix: np.ndarray
    u_annot_classes: List[str]
    u_decl_classes: List[str]
    num_decls: np.ndarray
    num_annots: np.ndarray
    im_ax: mpl.image.AxesImage = None        # AxesImage instance
    ax: mpl.axes.Axes = None                 # Axes instance (im_ax.axes)
    fig: mpl.figure.Figure = None            # Figure instance (im_ax.figure)
    text_annots: List[mpl.text.Text] = None  # Text annotations for each cell


def confusion_matrix(scorer: Scorer, threshold=-math.inf,
                     normalize=False):
    """
    Perform confusion matrix calculations and return relevant data.

    Args:
        scorer (atlas_scorer.models.Scorer): Scorer instance used for confusion
            matrix calculations
        threshold (float, optional): Confidence threshold to use for confusion
            matrix calculations; Default is **-math.inf**
        normalize (bool, optional): Whether to normalize each row of confusion
            matrix by the number of ground-truth annotations for the class. The
            last row that corresponds to false alarms is normalized such that
            values within the row correspond to FA/frame; Default **False**

    Returns:
        (ConfusionMatrixResult): A light-weight class containing the following
            attributes:
                normalize (bool): Whether the confusion matrix is normalized
                threshold (float): The confidence threshold used to generate
                    the confusion matrix
                count_matrix (np.ndarray): Normalized/unnormalized count-matrix
                    where rows correspond to ground-truth annotations and
                    columns correspond to declarations
                u_annot_classes (List[str]): List of annotation class names
                u_decl_classes (List[str]): List of declaration class names
                num_decls (np.ndarray): [N x 1] array containing number of decls
                    for each class in ``u_decl_classes``
                num_annots (np.ndarray): [N x 1] array containing number of
                    annots for each class in ``u_annot_classes``

    """
    # Detection link matrix (indexed by annot ind)
    link_matrix = scorer.det_link_map
    # Construct linkmatrix inverse mapping (indexed by decl ind)
    link_matrix_decl = OrderedDict()
    for annot_ind in link_matrix:
        for decl_ind in link_matrix[annot_ind]:
            if decl_ind not in link_matrix_decl:
                link_matrix_decl[decl_ind] = []
            link_matrix_decl[decl_ind].append(annot_ind)

    u_decl_classes = sorted(set(scorer.declarations_table['class'])) + ['N/A']
    u_annot_classes = sorted(set(scorer.annotations_table['class'])) + ['N/A']

    valid_decls = scorer.declarations_table['confidence'] > threshold
    # Index for decls into declaration table
    valid_decls_inds = valid_decls[valid_decls].index
    valid_decls_inds = valid_decls[valid_decls].index
    valid_decls_inds_set = set(valid_decls_inds)

    ignore_annots = scorer.annotations_table.apply(
        scorer.annotation_ignorable_function, axis=1, result_type='reduce')

    count_matrix = np.full((len(u_annot_classes), len(u_decl_classes)), np.nan)
    n_decls = np.full((len(u_decl_classes), 1), np.nan)
    n_annots = np.full((len(u_annot_classes), 1), np.nan)

    for i_decl_class, decl_class in enumerate(u_decl_classes):
        decl_indicator = (scorer.declarations_table['class'] == decl_class) & \
                         valid_decls
        # Index for decls into declaration table
        decl_indicator_inds = decl_indicator[decl_indicator].index

        n_decls[i_decl_class] = decl_indicator.sum()

        for i_annot_class, annot_class in enumerate(u_annot_classes):
            if annot_class == 'N/A':
                # Count all decls that are on NO annotations
                n_annots[i_annot_class] = np.nan
                count_decls_no_annots = 0
                for decl_indicator_ind in decl_indicator_inds:
                    annot_links = link_matrix_decl.get(decl_indicator_ind, None)
                    if (annot_links is None) or (len(annot_links) == 0):
                        count_decls_no_annots += 1
                count_matrix[i_annot_class, i_decl_class] = count_decls_no_annots

            elif decl_class == 'N/A':
                # Count all the annotations with NO declarations; ignore
                # ignorable annotations
                n_decls[i_decl_class] = np.nan
                annot_indicator = \
                    (scorer.annotations_table['class'] == annot_class) \
                    & ~ignore_annots
                annot_indicator_inds = annot_indicator[annot_indicator].index
                count_annots_no_decls = 0
                for annot_indicator_ind in annot_indicator_inds:
                    if ((annot_indicator_ind not in link_matrix) or
                            len(set(link_matrix[annot_indicator_ind]).
                                    intersection(valid_decls_inds_set)) == 0):
                        count_annots_no_decls += 1
                count_matrix[i_annot_class, i_decl_class] = \
                    count_annots_no_decls

            else:
                # Count all the annotations that have AT LEAST ONE declaration
                # on them; ignore ignorable annotations
                annot_indicator = \
                    (scorer.annotations_table['class'] == annot_class) \
                    & ~ignore_annots
                annot_indicator_inds = annot_indicator[annot_indicator].index
                n_annots[i_annot_class] = annot_indicator.sum()

                count_annots_with_decls = 0
                for annot_indicator_ind in annot_indicator_inds:
                    if annot_indicator_ind in link_matrix:
                        if len(set(link_matrix[annot_indicator_ind]).
                                intersection(decl_indicator_inds)) > 0:
                            count_annots_with_decls += 1
                count_matrix[i_annot_class, i_decl_class] = \
                    count_annots_with_decls

    if normalize:
        count_matrix[0:-1, :] = count_matrix[0:-1, :] / n_annots[0:-1]
        count_matrix[-1, :] = count_matrix[-1, :] / scorer.num_frames

    # Store relevant confusion matrix outputs for returning
    ret_val = ConfusionMatrixResult(
        normalize=normalize, threshold=threshold,
        count_matrix=count_matrix, u_annot_classes=u_annot_classes,
        u_decl_classes=u_decl_classes, num_decls=n_decls,
        num_annots=n_annots)

    return ret_val
