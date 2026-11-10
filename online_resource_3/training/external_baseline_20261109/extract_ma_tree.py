"""Extract the unchanged author MA-DT definitions, avoiding unused neural dependencies."""
import ast, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SRC=HERE/'vendor/malearn/malearn/estimators'
header='''# Exact source-definition subset of Healthy-AI/malearn (see extraction_manifest.json).
# Original license is retained in vendor/malearn/LICENSE.md.
import math, copy, warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, is_classifier, clone, _fit_context
from sklearn.utils import check_random_state, Bunch
from sklearn.utils.multiclass import unique_labels
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.tree import _tree
from sklearn.tree._tree import Tree, _build_pruned_tree_ccp, ccp_pruning_path
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _check_sample_weight
'''
spec={'tree.py':None,'criterion.py':None,'missingness_utils.py':['check_missingness_mask'],
      'base.py':['ClassifierMixin','BaseMADT'],'classification.py':['MADTClassifier']}
parts=[header]; manifest=[]
for file,names in spec.items():
    source=(SRC/file).read_text(); tree=ast.parse(source)
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.ClassDef)):continue
        if names is not None and node.name not in names:continue
        piece=ast.get_source_segment(source,node)
        assert ast.dump(ast.parse(piece).body[0])==ast.dump(node)
        parts.append(piece)
        manifest.append({'source':file,'definition':node.name,'line':node.lineno,'end_line':node.end_lineno,'sha256':hashlib.sha256(piece.encode()).hexdigest(),'ast_identical':True})
parts.insert(-1,'CRITERIA_CLF = {"info_gain": info_gain_scorer, "gini": gini_scorer}')
out='\n\n'.join(parts)+'\n'
(HERE/'ma_tree.py').write_text(out)
(HERE/'extraction_manifest.json').write_text(json.dumps({'commit':'fbbdd0ca2c9d0d8072f49d82ad35f4391098863f','definitions':manifest,'generated_sha256':hashlib.sha256(out.encode()).hexdigest(),'changes':'Only imports/module layout; all extracted scientific definitions are AST-identical to author source. No model logic changes.'},indent=2))
print(len(manifest),'unchanged definitions extracted')
