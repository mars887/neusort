import argparse
import re
import sys
import textwrap
from typing import List, Tuple, Dict, Any

import torch
from config import Config
from logger import CustomLogger, LogLevel


# CLI schema (single source of truth)
# - tags: TaskStarter, FinalTaskStarter, Task, Param, NonTerminal (unused here)
# - subparams: dict of logical sub-keys for grouped syntax, each with names/type/dest
CLI_SPEC: List[Dict[str, Any]] = [
    {
        "names": ["--cluster"],
        "action": "store_true",
        "help": "Group images into clusters based on feature distance threshold.",
        "tags": ["TaskStarter"],
        "subparams": {
            "threshold": {"names": ["threshold"], "type": "float", "dest": "cluster_threshold"},
            "similarity_percent": {"names": ["similarity_percent", "percent"], "type": "float", "dest": "cluster_similarity_percent"},
            "cluster_min_size": {"names": ["cluster_min_size", "min_size"], "type": "int", "dest": "cluster_min_size"},
            "cluster_naming_mode": {"names": ["cluster_naming_mode", "naming_mode", "mode"], "type": "str", "dest": "cluster_naming_mode"},
            "save_discarded": {"names": ["save_discarded"], "type": "bool", "dest": "save_discarded"},
            "save_mode": {"names": ["save_mode"], "type": "str", "dest": "save_mode"},
            "algorithm": {"names": ["algorithm", "cluster_algorithm"], "type": "str", "dest": "algorithm"},
            "cluster_pca": {"names": ["pca", "cluster_pca"], "type": "bool", "dest": "cluster_pca"},
        },
    },
    {
        "names": ["--sorting"],
        "action": "store_true",
        "help": "Run sorting pipeline.",
        "tags": ["TaskStarter"],
        "subparams": {
            "lookahead": {"names": ["lookahead"], "type": "int", "dest": "lookahead"},
            "neighbors_k_limit": {"names": ["neighbors_k_limit"], "type": "int", "dest": "neighbors_k_limit"},
            "two_opt_shift": {"names": ["two_opt_shift"], "type": "int", "dest": "two_opt_shift"},
            "two_opt_block_size": {"names": ["two_opt_block_size"], "type": "int", "dest": "two_opt_block_size"},
            "sort_optimizer": {"names": ["sort_optimizer"], "type": "str", "dest": "sort_optimizer"},
            "sort_strategy": {"names": ["sort_strategy", "strategy"], "type": "str", "dest": "sort_strategy"},
            "list_only": {"names": ["list_only"], "type": "bool", "dest": "list_only"},
        },
    },
    {
        "names": ["--find"],
        "nargs": "?",
        "const": "__PIPE__",
        "help": "Search mode (optional path or pipeline).",
        "tags": ["FinalTaskStarter"],
        "accept_value": True,
        "subparams": {
            "batch_size": {"names": ["batch_size"], "type": "int", "dest": "batch_size"},
            "find_neighbors": {"names": ["find_neighbors", "neighbors", "k"], "type": "int", "dest": "find_neighbors"},
            "find_result_type": {"names": ["find_result_type", "result_type", "format"], "type": "str", "dest": "find_result_type"},
            "tsv_neighbors": {"names": ["tsv_neighbors"], "type": "int", "dest": "tsv_neighbors"},
        },
    },
    # Files / model / misc / db
    {"names": ["-m", "--model", "--model_name"], "type": str, "help": "Model name.", "tags": ["Param"]},
    {"names": ["--more_scan"], "action": "store_true", "help": "Extended input scan.", "tags": ["Param"]},
    {"names": ["--feature_workers"], "type": int, "help": "Workers for feature extraction.", "tags": ["Param"]},
    {"names": ["--image_batch_size"], "type": int, "help": "Batch size for feature extraction.", "tags": ["Param"]},
    {"names": ["--input", "--input_folder", "-i"], "type": str, "help": "Input folder.", "tags": ["Param"]},
    {"names": ["--output", "--output_folder", "-o"], "type": str, "help": "Output folder.", "tags": ["Param"]},
    {"names": ["--index_file"], "type": str, "help": "FAISS index file.", "tags": ["Param"]},
    {"names": ["--out_tsv"], "type": str, "help": "Neighbors TSV output.", "tags": ["Param"]},
    {"names": ["--use_cpu", "--cpu"], "action": "store_true", "help": "Force CPU.", "tags": ["Param"]},
    {"names": ["--loglevel"], "type": str, "choices": ["default", "error", "quiet", "debug"], "help": "Log level.", "tags": ["Param"]},
    {"names": ["--list_objects"], "action": "store_true", "help": "List DB file paths.", "tags": ["Task"]},
    {"names": ["--move_db"], "type": str, "nargs": 2, "help": "Move DB entries OLD_ROOT NEW_ROOT.", "tags": ["Task"]},
    {"names": ["--print_params"], "type": str, "choices": ["all", "entered"], "help": "Print parameters before run.", "tags": ["Param"]},
]


def _first_long_name(names: List[str]) -> str:
    for n in names:
        if n.startswith("--"):
            return n
    return names[0]


# Build indices from schema
TOP_NAME_TO_SPEC: Dict[str, Dict[str, Any]] = {}
CLI_TOP_SPECS: List[Dict[str, Any]] = []
PARENT_SUBPARAM_MAP: Dict[str, Dict[str, Tuple[str, str]]] = {}

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
                submap[alias] = (dest, typ)
        PARENT_SUBPARAM_MAP[parent] = submap


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
    for item in _split_group_items(group_token):
        if "=" in item:
            k, v = item.split("=", 1)
        else:
            k, v = item, ""
        k = k.strip().lower()
        v = v.strip()
        if k not in spec:
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
        setattr(args, dest, val)
        # record dotted for reporting
        parent_key = parent_flag.lstrip('-')
        subname = dest
        if subname.startswith(parent_key + "_"):
            subname = subname[len(parent_key) + 1 :]
        updates[f"{parent_key}.{subname}"] = val
    return updates


# Build parser from top-level specs (SubParams are not added to argparse)
parser = argparse.ArgumentParser(
    description="Neusort CLI",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent(
        """Examples:
  --cluster algorithm=dbscan:threshold=0.3
  --sorting lookahead=4:sort_optimizer=2opt
  --find path/to/query.jpg find_result_type=both
  --move_db OLD_ROOT NEW_ROOT --list_objects
"""
    ),
)

for spec in CLI_TOP_SPECS:
    names = spec["names"]
    kwargs: Dict[str, Any] = {}
    if "help" in spec:
        kwargs["help"] = spec["help"]
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

if getattr(args, "print_params", None) in ("all", "entered"):
    try:
        if args.print_params == "entered":
            for dk, dv in ENTERED_GROUPED.items():
                print(f"{dk}={dv}")
            for k, v in ENTERED_PARAMS.items():
                attr = k.lstrip('-').replace('-', '_')
                print(f"{attr}={v}")
        else:
            attrs = set()
            # known top-level attrs
            for spec in CLI_TOP_SPECS:
                for nm in spec["names"]:
                    attrs.add(nm.lstrip('-').replace('-', '_'))
            # known subparam dests
            for submap in PARENT_SUBPARAM_MAP.values():
                for dest, _t in submap.values():
                    attrs.add(dest)
            for attr in sorted(attrs):
                v = getattr(args, attr, None)
                if v is not None and v is not False:
                    print(f"{attr}={v}")
    except Exception:
        pass


# Ensure presence of expected attributes for Config
if not hasattr(args, "list_only"):
    setattr(args, "list_only", False)

LOGGER = CustomLogger(level=LogLevel(getattr(args, "loglevel", "default")))
CONFIG = Config(args)


if CONFIG.model.use_cpu:
    DEVICE = torch.device("cpu")
    LOGGER.info("Using CPU for model inference")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
      NonTerminal     -> Internal (not exposed; usually defaults live in code)
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
   - Defaults should live in Config (not in CLI_SPEC) to avoid duplication.

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
