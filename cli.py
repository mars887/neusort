import argparse
import os
import re
import sys
import textwrap
from typing import List, Tuple, Dict, Any

import torch
from config import Config
from logger import CustomLogger, LogLevel
from models import MODEL_CONFIGS, set_model_logger
from model_factory import set_model_factory_runtime
import runtime_state as runtime


DEFAULT_FEATURE_WORKERS = max(1, os.cpu_count() or 1)


# CLI schema
# - tags: TaskStarter, FinalTaskStarter, Task, Param, NonTerminal
# - subparams: dict of logical sub-keys for grouped syntax, each with names/type/dest
CLI_SPEC: List[Dict[str, Any]] = [
    {
        "names": ["--cluster"],
        "action": "store_true",
        "help": "Cluster feature vectors and optionally export/save groupings.",
        "tags": ["TaskStarter"],
        "subparams": {
            "threshold": {
                "names": ["threshold"],
                "type": "float",
                "dest": "cluster_threshold",
                "default": 0.35,
                "help": "Distance threshold for distance-based algorithms.",
            },
            "similarity_percent": {
                "names": ["similarity_percent", "percent"],
                "type": "float",
                "dest": "cluster_similarity_percent",
                "default": 50.0,
                "help": "Percent of closest neighbors kept when building similarity graphs.",
            },
            "cluster_min_size": {
                "names": ["cluster_min_size", "min_size"],
                "type": "int",
                "dest": "cluster_min_size",
                "default": 2,
                "help": "Discard clusters smaller than this size.",
            },
            "cluster_naming_mode": {
                "names": ["cluster_naming_mode", "naming_mode", "mode"],
                "type": "str",
                "dest": "cluster_naming_mode",
                "default": "default",
                "choices": ["default", "distance", "distance_plus"],
                "help": "Choose how cluster folders are named.",
            },
            "save_discarded": {
                "names": ["save_discarded"],
                "type": "bool",
                "dest": "save_discarded",
                "default": True,
                "help": "Store images that were not assigned to any cluster.",
            },
            "save_mode": {
                "names": ["save_mode"],
                "type": "str",
                "dest": "save_mode",
                "default": "default",
                "choices": ["default", "json", "print", "group_filling"],
                "help": "Select how to persist clustering results.",
            },
            "algorithm": {
                "names": ["algorithm", "cluster_algorithm"],
                "type": "str",
                "dest": "algorithm",
                "default": "distance",
                "choices": ["distance", "hdbscan", "dbscan", "cc_graph", "mutual_graph", "agglomerative", "agglomerative_complete", "optics", "snn", "rank_mutual", "adaptive_graph"],
                "help": "Clustering backend.",
            },
            "cluster_pca": {
                "names": ["pca", "cluster_pca"],
                "type": "bool",
                "dest": "cluster_pca",
                "default": False,
                "help": "Run PCA/whitening before clustering to denoise features.",
            },
            "group_filling_size": {
                "names": ["group_filling_size", "group_size", "x"],
                "type": "int",
                "dest": "group_filling_size",
                "default": 10,
                "help": "Target group size when save_mode=group_filling.",
            },
            "cluster_splitting_mode": {
                "names": ["cluster_splitting_mode", "split_mode"],
                "type": "str",
                "dest": "cluster_splitting_mode",
                "default": "recluster",
                "choices": ["recluster", "fi"],
                "help": "How to split over-sized groups when filling.",
            },
            "similar_fill": {
                "names": ["similar_fill", "similar_filling"],
                "type": "bool",
                "dest": "cluster_similar_fill",
                "default": False,
                "help": "Prefer fillers that are closest to the current group centroid.",
            },
            "enable_refine": {
                "names": ["enable_refine", "refine_clusters"],
                "type": "bool",
                "dest": "cluster_enable_refine",
                "default": False,
                "help": "Run post-cluster refinement (split/prune/garbage filter).",
            },
        },
    },
    {
        "names": ["--sorting"],
        "action": "store_true",
        "help": "Run the MST sorting pipeline to arrange photos.",
        "tags": ["TaskStarter"],
        "subparams": {
            "lookahead": {
                "names": ["lookahead"],
                "type": "int",
                "dest": "lookahead",
                "default": 100,
                "help": "DFS lookahead depth for main-component traversal.",
            },
            "neighbors_k_limit": {
                "names": ["neighbors_k_limit"],
                "type": "int",
                "dest": "neighbors_k_limit",
                "default": 1024,
                "help": "Limit of nearest neighbors kept for MST construction.",
            },
            "two_opt_shift": {
                "names": ["two_opt_shift"],
                "type": "int",
                "dest": "two_opt_shift",
                "default": 90,
                "help": "Shift window used by the 2-opt local optimizer.",
            },
            "two_opt_block_size": {
                "names": ["two_opt_block_size"],
                "type": "int",
                "dest": "two_opt_block_size",
                "default": 100,
                "help": "How many nodes are processed per 2-opt block.",
            },
            "sort_optimizer": {
                "names": ["sort_optimizer"],
                "type": "str",
                "dest": "sort_optimizer",
                "default": "2opt",
                "help": "Local optimizer applied after the main traversal.",
            },
            "sort_strategy": {
                "names": ["sort_strategy", "strategy"],
                "type": "str",
                "dest": "sort_strategy",
                "default": "farthest_insertion",
                "choices": ["dfs", "farthest_insertion"],
                "help": "Traversal strategy for the main MST component.",
            },
        },
    },
    {
        "names": ["--index_only"],
        "action": "store_true",
        "help": "Only compute/cache features (build the index) and exit; skips sorting/clustering/search.",
        "tags": ["TaskStarter"],
    },
    {
        "names": ["--find"],
        "nargs": "?",
        "const": "__PIPE__",
        "default": None,
        "metavar": "[QUERY]",
        "help": "Search for nearest images (path argument or pipeline input).",
        "tags": ["FinalTaskStarter"],
        "subparams": {
            "batch_size": {
                "names": ["batch_size"],
                "type": "int",
                "dest": "batch_size",
                "default": 2048,
                "help": "Batch size for k-NN distance evaluations.",
            },
            "find_neighbors": {
                "names": ["find_neighbors", "neighbors", "k"],
                "type": "int",
                "dest": "find_neighbors",
                "default": 5,
                "help": "How many neighbors to print for each query.",
            },
            "find_result_type": {
                "names": ["find_result_type", "result_type", "format"],
                "type": "str",
                "dest": "find_result_type",
                "default": "both",
                "choices": ["indexed", "path", "both"],
                "help": "Output style: FAISS positions, filesystem paths, or both.",
            },
            "tsv_neighbors": {
                "names": ["tsv_neighbors"],
                "type": "int",
                "dest": "tsv_neighbors",
                "default": 5,
                "help": "Neighbors per row when exporting TSV summaries.",
            },
            "backend": {
                "names": ["backend"],
                "type": "str",
                "dest": "backend",
                "default": "auto",
                "choices": ["auto", "regnet", "clip"],
                "help": "Force the embedding backend (auto picks based on model).",
            },
            "query_mode": {
                "names": ["query_mode", "mode"],
                "type": "str",
                "dest": "query_mode",
                "default": "image",
                "choices": ["image", "text", "image+text"],
                "help": "Type of query consumed by --find.",
            },
            "fusion_mode": {
                "names": ["fusion_mode", "fusion"],
                "type": "str",
                "dest": "fusion_mode",
                "default": "simple",
                "choices": ["simple", "directional"],
                "help": "How image/text embeddings are fused in image+text mode.",
            },
            "image_weight": {
                "names": ["image_weight", "w_img"],
                "type": "float",
                "dest": "image_weight",
                "default": 0.5,
                "help": "Weight of the image branch for fusion queries.",
            },
            "text_weight": {
                "names": ["text_weight", "w_txt"],
                "type": "float",
                "dest": "text_weight",
                "default": 0.5,
                "help": "Weight of the text branch for fusion queries.",
            },
            "directional_alpha": {
                "names": ["directional_alpha", "alpha"],
                "type": "float",
                "dest": "directional_alpha",
                "default": 0.7,
                "help": "Alpha used by directional fusion mode.",
            },
            "base_prompt": {
                "names": ["base_prompt", "base_text"],
                "type": "str",
                "dest": "base_prompt",
                "default": "a photo on a road",
                "help": "Default textual hint for directional CLIP queries.",
            },
        },
    },
    # Files / model / misc / db
    {
        "names": ["-m", "--model", "--model_name"],
        "type": str,
        "default": "clip_vit_liaon",
        "metavar": "MODEL",
        "help": "Backbone used for feature extraction (see Supported models section).",
        "tags": ["Param"],
    },
    {
        "names": ["--more_scan"],
        "action": "store_true",
        "default": False,
        "help": "Use multi-crop feature extraction instead of a single center crop.",
        "tags": ["Param"],
    },
    {
        "names": ["--feature_workers"],
        "type": int,
        "default": DEFAULT_FEATURE_WORKERS,
        "metavar": "N",
        "help": "How many worker processes are used while encoding features.",
        "tags": ["Param"],
    },
    {
        "names": ["--image_batch_size"],
        "type": int,
        "default": 1024,
        "metavar": "BATCH",
        "help": "Mini-batch size for the feature extractor.",
        "tags": ["Param"],
    },
    {
        "names": ["--input", "--input_folder", "-i"],
        "type": str,
        "default": "input_images",
        "metavar": "DIR",
        "help": "Folder with unsorted images that will be processed.",
        "tags": ["Param"],
    },
    {
        "names": ["--output", "--output_folder", "-o"],
        "type": str,
        "default": "sorted_images",
        "metavar": "DIR",
        "help": "Folder where sorted images (or lists) will be written.",
        "tags": ["Param"],
    },
    {
        "names": ["--index_file"],
        "type": str,
        "default": "faiss.index",
        "metavar": "PATH",
        "help": "FAISS index path used for clustering/search.",
        "tags": ["Param"],
    },
    {
        "names": ["--out_tsv"],
        "type": str,
        "default": "neighbor_list.tsv",
        "metavar": "PATH",
        "help": "Where to store neighbor summaries in TSV format.",
        "tags": ["Param"],
    },
    {
        "names": ["--use_cpu", "--cpu"],
        "action": "store_true",
        "default": False,
        "help": "Force CPU execution even if a GPU is visible.",
        "tags": ["Param"],
    },
    {
        "names": ["--loglevel"],
        "type": str,
        "default": "default",
        "choices": ["default", "error", "quiet", "debug"],
        "metavar": "LEVEL",
        "help": "Verbosity level for console logging.",
        "tags": ["Param"],
    },
    {
        "names": ["--list_objects"],
        "action": "store_true",
        "default": False,
        "help": "Print every file path stored in the SQLite feature DB.",
        "tags": ["Task"],
    },
    {
        "names": ["--move_db"],
        "type": str,
        "nargs": 2,
        "metavar": ("OLD_ROOT", "NEW_ROOT"),
        "help": "Rewrite database paths by replacing OLD_ROOT with NEW_ROOT.",
        "tags": ["Task"],
    },
    {
        "names": ["--print_params"],
        "type": str,
        "choices": ["all", "entered"],
        "metavar": "MODE",
        "help": "Dump effective parameters (all) or only values provided via CLI (entered).",
        "tags": ["Param"],
    },
    {
        "names": ["--list_only"],
        "action": "store_true",
        "default": False,
        "help": "Skip file system writes and only print the operations that would run.",
        "tags": ["Param"],
    },
    {
        "names": ["--query_text"],
        "type": str,
        "metavar": "TEXT",
        "help": "Inline text used by --find text/image+text modes.",
        "tags": ["Param"],
    },
]


def _build_supported_models_help() -> str:
    if not MODEL_CONFIGS:
        return "  (no models registered)"
    longest_name = max(len(name) for name in MODEL_CONFIGS)
    lines = []
    for name in sorted(MODEL_CONFIGS):
        flops = MODEL_CONFIGS[name].get("flops") or "n/a"
        lines.append(f"  {name.ljust(longest_name)}  {flops}")
    return "\n".join(lines)


SUPPORTED_MODELS_HELP = _build_supported_models_help()


def _build_help_epilog() -> str:
    examples = textwrap.dedent(
        """Examples:
  --cluster algorithm=dbscan:threshold=0.3
  --sorting lookahead=4:sort_optimizer=2opt
  --find path/to/query.jpg find_result_type=both
  --move_db OLD_ROOT NEW_ROOT --list_objects
"""
    ).rstrip()
    return f"{examples}\n\nSupported models (FLOPs):\n{SUPPORTED_MODELS_HELP}"


PARSER_EPILOG = _build_help_epilog()


class NeusortHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Allow multiline help text while also appending default values automatically."""


def _first_long_name(names: List[str]) -> str:
    for n in names:
        if n.startswith("--"):
            return n
    return names[0]


# Build indices from schema
TOP_NAME_TO_SPEC: Dict[str, Dict[str, Any]] = {}
CLI_TOP_SPECS: List[Dict[str, Any]] = []
PARENT_SUBPARAM_MAP: Dict[str, Dict[str, Tuple[str, str]]] = {}
PARENT_VALUE_DEST: Dict[str, str] = {}
DEFAULT_GROUPED_VALUES: Dict[str, Any] = {}


def _arg_dest_from_names(names: List[str]) -> str:
    for n in names:
        if n.startswith("--"):
            base = n.lstrip("-")
            break
    else:
        base = names[0].lstrip("-")
    return base.replace("-", "_")

def _format_subparam_help(subparams: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for item in subparams.values():
        alias = ", ".join(item["names"])
        parts = [f"  {alias}: {item['help']}"]
        if item.get("choices"):
            parts.append(f"Choices: {', '.join(item['choices'])}.")
        if "default" in item:
            parts.append(f"Default: {item['default']}.")
        lines.append(" ".join(parts))
    return "\nSub-parameters:\n" + "\n".join(lines)


def _subparam_dotted_key(parent_flag: str, dest: str) -> str:
    parent_key = parent_flag.lstrip("-")
    subname = dest
    if subname.startswith(parent_key + "_"):
        subname = subname[len(parent_key) + 1 :]
    return f"{parent_key}.{subname}"


for spec in CLI_SPEC:
    # record top-level names
    CLI_TOP_SPECS.append(spec)
    for nm in spec["names"]:
        TOP_NAME_TO_SPEC[nm] = spec
    # record subparams by parent canonical flag
    subparams = spec.get("subparams") or {}
    if subparams:
        parent = _first_long_name(spec["names"])
        submap: Dict[str, Tuple[str, str]] = {}
        for _, s in subparams.items():
            dest = s["dest"]
            typ = s.get("type", "str")
            for alias in s["names"]:
                submap[alias.lower()] = (dest, typ)
            dotted = _subparam_dotted_key(parent, dest)
            if "default" in s:
                DEFAULT_GROUPED_VALUES[dotted] = s["default"]
        PARENT_SUBPARAM_MAP[parent] = submap
        PARENT_VALUE_DEST[parent] = _arg_dest_from_names(spec["names"])


GROUP_VALUE_SPLIT_PATTERN = re.compile(r":(?=[A-Za-z0-9_\-]+(?:=|$))")
TRUE_STRINGS = {"1", "true", "yes", "on"}
FALSE_STRINGS = {"0", "false", "no", "off"}


def _fail_group_argument(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.exit(2)


def _split_group_items(raw: str) -> List[str]:
    return [item.strip() for item in GROUP_VALUE_SPLIT_PATTERN.split(raw) if item.strip()]


def _parse_bool(value: str) -> bool:
    if value == "" or value is None:
        return True
    lowered = value.lower()
    if lowered in TRUE_STRINGS:
        return True
    if lowered in FALSE_STRINGS:
        return False
    _fail_group_argument(f"Unsupported boolean value '{value}'. Use one of {sorted(TRUE_STRINGS | FALSE_STRINGS)}.")
    return False


def _apply_group(parent_flag: str, group_token: str, args: argparse.Namespace) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    spec = PARENT_SUBPARAM_MAP.get(parent_flag, {})
    parent_dest = PARENT_VALUE_DEST.get(parent_flag)
    parent_value_set = False
    for idx, item in enumerate(_split_group_items(group_token)):
        raw_key = item
        raw_val = ""
        if "=" in item:
            raw_key, raw_val = item.split("=", 1)
        key_clean = raw_key.strip()
        k = key_clean.lower()
        v = raw_val.strip()
        if k not in spec:
            if (
                not parent_value_set
                and idx == 0
                and "=" not in item
                and parent_dest
            ):
                setattr(args, parent_dest, key_clean)
                parent_value_set = True
                continue
            _fail_group_argument(f"Unknown sub-argument '{k}' for {parent_flag}.")
        dest, typ = spec[k]
        if typ == "int":
            val = int(v)
        elif typ == "float":
            val = float(v)
        elif typ == "bool":
            val = True if v == "" else _parse_bool(v)
        else:
            val = v
        # record dotted for reporting
        updates[_subparam_dotted_key(parent_flag, dest)] = val
    return updates


# Build parser from top-level specs (SubParams are not added to argparse)
parser = argparse.ArgumentParser(
    description="Neusort CLI",
    formatter_class=NeusortHelpFormatter,
    epilog=PARSER_EPILOG,
)

for spec in CLI_TOP_SPECS:
    names = spec["names"]
    kwargs: Dict[str, Any] = {}
    help_text = spec.get("help")
    if help_text and spec.get("subparams"):
        help_text = help_text.rstrip() + "\n" + _format_subparam_help(spec["subparams"])
    if help_text:
        kwargs["help"] = help_text
    if "action" in spec:
        kwargs["action"] = spec["action"]
    if "type" in spec:
        kwargs["type"] = spec["type"]
    if "nargs" in spec:
        kwargs["nargs"] = spec["nargs"]
    if "const" in spec:
        kwargs["const"] = spec["const"]
    if "choices" in spec:
        kwargs["choices"] = spec["choices"]
    if "metavar" in spec:
        kwargs["metavar"] = spec["metavar"]
    if "default" in spec:
        kwargs["default"] = spec["default"]
    parser.add_argument(*names, **kwargs)


# Parse args and grouped subparams
raw_cli_args = sys.argv[1:]
args, _unknown = parser.parse_known_args(raw_cli_args)

ENTERED_GROUPED: Dict[str, Any] = {}
i = 0
while i < len(raw_cli_args):
    tok = raw_cli_args[i]
    spec = TOP_NAME_TO_SPEC.get(tok)
    if not spec:
        i += 1
        continue
    nxt = raw_cli_args[i + 1] if (i + 1) < len(raw_cli_args) else None
    if nxt is None:
        i += 1
        continue
    # handle direct value after --find when no '=' or ':' present
    if tok == "--find" and not nxt.startswith("-") and ("=" not in nxt and ":" not in nxt):
        setattr(args, "find", nxt)
        i += 2
        continue
    # grouped subparams value
    if not nxt.startswith("-") and ("subparams" in spec and spec["subparams"]):
        updates = _apply_group(tok, nxt, args)
        ENTERED_GROUPED.update(updates)
        i += 2
        continue
    i += 1


# Build task plan and heavy tasks
CLI_TAGGED = []
for spec in CLI_SPEC:
    tagset = set(spec.get("tags", []))
    if "TaskStarter" in tagset:
        CLI_TAGGED.append((_first_long_name(spec["names"]), "TaskStarter"))
    if "FinalTaskStarter" in tagset:
        CLI_TAGGED.append((_first_long_name(spec["names"]), "FinalTaskStarter"))
    if "Task" in tagset:
        CLI_TAGGED.append((_first_long_name(spec["names"]), "Task"))

TASK_STARTERS = {name for (name, tag) in CLI_TAGGED if tag == "TaskStarter"}
FINAL_TASK_STARTERS = [name for (name, tag) in CLI_TAGGED if tag == "FinalTaskStarter"]
TASKS = {name for (name, tag) in CLI_TAGGED if tag == "Task"}
HEAVY_TASKS = set(TASK_STARTERS) | set(FINAL_TASK_STARTERS)

task_plan: List[str] = []
final_seen: List[str] = []
for tok in raw_cli_args:
    if tok in TASKS or tok in TASK_STARTERS or tok in FINAL_TASK_STARTERS:
        if tok in FINAL_TASK_STARTERS:
            final_seen.append(tok)
        else:
            task_plan.append(tok)

if final_seen:
    task_plan.append(final_seen[0])

if not task_plan:
    task_plan = ["--sorting"]

TASK_PLAN = task_plan


def _collect_entered_params(tokens: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("-"):
            i += 1
            continue
        val: Any = True
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            val = tokens[i + 1]
            i += 1
        out[t] = val
        i += 1
    return out


ENTERED_PARAMS = _collect_entered_params(raw_cli_args)



LOGGER = CustomLogger(level=LogLevel(getattr(args, "loglevel", None) or "default"))
set_model_logger(LOGGER)
set_model_factory_runtime(logger=LOGGER)
runtime.set_runtime(logger=LOGGER)
# Merge grouped defaults declared in CLI_SPEC with user-provided overrides
EFFECTIVE_GROUPED: Dict[str, Any] = dict(DEFAULT_GROUPED_VALUES)
EFFECTIVE_GROUPED.update(ENTERED_GROUPED)
# Pass grouped sub-params separately to Config so that subparam namespaces
# remain isolated per parent flag and are not read from argparse attrs.
CONFIG = Config(args, EFFECTIVE_GROUPED)


if CONFIG.model.use_cpu:
    DEVICE = torch.device("cpu")
    LOGGER.info("Using CPU for model inference")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_model_factory_runtime(device=DEVICE)
runtime.set_runtime(device=DEVICE)
    
"""
Neusort CLI — Developer Guide

Overview
- The CLI is defined by a single schema (CLI_SPEC) in this module.
- Only top-level parameters are registered with argparse. SubParams are parsed
  manually from grouped values that follow their parent flag.
- Task execution order is derived from parameter tags; the first FinalTaskStarter
  (e.g., --find) is executed last and excludes subsequent tasks.

Key Concepts
- Top-level parameter: a regular CLI option (Param, Task, TaskStarter, FinalTaskStarter)
  declared directly in CLI_SPEC[...]['names'] and added to argparse.
- SubParam: a logical child option that belongs to a specific parent TaskStarter;
  it is not added to argparse. SubParams are passed as a single grouped value
  placed after the parent flag using key=value fragments separated by ':'.

  Example: --cluster algorithm=dbscan:threshold=0.3:min_size=3

Schema Structure (CLI_SPEC)
- Each entry is a dict with:
  - names: list[str] — aliases, first long name is canonical (e.g., ["--cluster"]).
  - help: str — human-readable help.
  - tags: list[str] — one or more of:
      Param           -> Plain parameter
      Task            -> Lightweight task (can be repeated)
      TaskStarter     -> Starts a pipeline task (e.g., sorting, clustering)
      FinalTaskStarter-> Must run last; if multiple, only the first is kept
      NonTerminal     -> Internal (not exposed; defaults live in code if needed)
  - action|type|nargs|choices|const|metavar — standard argparse options for
    top-level argument behavior.
  - subparams: dict[str, SubSpec] (optional) — keyed by a logical name; each SubSpec is:
      {
        "names": ["alias1", "alias2", ...],   # grouped aliases
        "type": "str"|"int"|"float"|"bool",  # value type in grouped syntax
        "dest": "config_attr_name"              # attribute assigned on args
      }

Grouped Syntax Rules
- SubParams are specified as a single token immediately following the parent flag.
  - Fragments are separated by ':' (colon). Each fragment is key, or key=value.
  - Types: int and float are coerced; bool accepts: empty (true), 1/true/yes/on, 0/false/no/off.
  - Aliases: SubSpec.names allows multiple keys to map to the same dest.
- Parent-specific rules:
  - --find supports a direct path value when no '=' or ':' is present.
  - Example: --find path/to.jpg find_result_type=both

Task Planning and Heavy Tasks
- TASK_PLAN is built from the raw token order, filtered by tags:
  - Task (e.g., --list_objects, --move_db) run where they appear.
  - TaskStarter (e.g., --sorting, --cluster) run where they appear.
  - FinalTaskStarter (e.g., --find) — only the first encountered is appended to the end.
- HEAVY_TASKS is computed from {TaskStarter} ∪ {FinalTaskStarter} and used in main.py
  to gate heavy preconditions (like verifying the input folder exists).

Adding a New Parameter
1) Top-level Param/Task/TaskStarter
   - Add a new entry to CLI_SPEC with its names/help/tags and argparse settings.
   - If it is a TaskStarter with grouped subparams, also add a subparams map.
2) SubParam for an existing TaskStarter
   - Add a new SubSpec to the parent entry's 'subparams' with:
     - names: list of grouped keys
     - type: one of str/int/float/bool
     - dest: args attribute to set (ensure Config reads this attribute)
3) Update Config if needed
   - Keep Config attribute names stable; set 'dest' to match what Config expects.
   - Declare defaults inside CLI_SPEC so --help stays truthful; Config receives
     merged default + user values via the grouped map.

Validation and Printing
- --print_params entered: prints grouped dotted keys (parent.subkey=value) and top-level
  flags that were explicitly set.
- --print_params all: prints all known non-empty values (top-level + subparam dests).

Examples
- Clustering:  --cluster algorithm=dbscan:threshold=0.3
- Search:      --find sample.jpg find_result_type=both
- Move+List:   --move_db "OLD_ROOT" "NEW_ROOT" --list_objects
- Sorting:     --sorting lookahead=4:sort_optimizer=2opt

Notes
- SubParams cannot be specified as top-level flags; they only apply via grouped syntax.
- If multiple FinalTaskStarter flags are given, only the first is executed; others are ignored.
- NonTerminal items are not exposed via CLI; they should be defaulted in code.
"""

# Optional parameters printing
if getattr(args, "print_params", None) in ("all", "entered"):
    try:
        if args.print_params == "entered":
            for dk, dv in ENTERED_GROUPED.items():
                print(f"{dk}={dv}")
            for k, v in ENTERED_PARAMS.items():
                attr = k.lstrip('-').replace('-', '_')
                print(f"{attr}={v}")
        else:
            # Top-level args that were set or have values
            attrs = set()
            for spec in CLI_TOP_SPECS:
                for nm in spec["names"]:
                    attrs.add(nm.lstrip('-').replace('-', '_'))
            for attr in sorted(attrs):
                v = getattr(args, attr, None)
                if v is not None and v is not False:
                    print(f"{attr}={v}")

            # Effective SubParams from Config, printed as dotted keys
            printed = set()
            for parent_flag, submap in PARENT_SUBPARAM_MAP.items():
                parent_key = parent_flag.lstrip('-')
                cfg_obj = None
                if parent_key == 'cluster':
                    cfg_obj = CONFIG.clustering
                elif parent_key == 'sorting':
                    cfg_obj = CONFIG.sorting
                elif parent_key == 'find':
                    cfg_obj = CONFIG.search
                if cfg_obj is None:
                    continue
                # unique dests for this parent
                dests = {dest for (dest, _t) in submap.values()}
                for dest in sorted(dests):
                    subname = dest
                    if subname.startswith(parent_key + "_"):
                        subname = subname[len(parent_key) + 1 :]
                    # map to config attribute name if needed
                    cfg_attr = subname
                    if parent_key == 'cluster':
                        if subname == 'pca':
                            cfg_attr = 'pca_enabled'
                        elif subname == 'splitting_mode':
                            cfg_attr = 'cluster_splitting_mode'
                    elif parent_key == 'sorting':
                        if subname == 'sort_optimizer':
                            cfg_attr = 'optimizer'
                        elif subname == 'sort_strategy':
                            cfg_attr = 'strategy'
                    elif parent_key == 'find' and subname == 'batch_size':
                        cfg_attr = 'global_knn_batch_size'
                    v = getattr(cfg_obj, cfg_attr, None)
                    key = f"{parent_key}.{subname}"
                    if key in printed:
                        continue
                    if v is not None and v is not False:
                        printed.add(key)
                        print(f"{key}={v}")
    except Exception:
        pass
