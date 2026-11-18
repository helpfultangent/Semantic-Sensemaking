#!/bin/bash
# Tapis entrypoint shared by every Semantic Sensemaking app.
#
# Tapis stages the job's file inputs into $_tapisExecSystemInputDir and archives
# $_tapisExecSystemOutputDir. Parameters defined as SSM_* environment variables in
# the app definition (e.g. SSM_CORPUS__MAX_FILES) are read by the library's
# configuration directly, so they need no handling here.
#
# Staged inputs recognized (all optional except the corpus):
#   corpus/         documents to analyze
#   lexicon.json    user-defined dictionary (JSON)
#   lexicon.csv     user-defined dictionary (CSV: group,term)
#   taxonomy.json   science backbone
#   job.yaml        full job configuration
set -euo pipefail

IN="${_tapisExecSystemInputDir:?Tapis input directory is not set}"
OUT="${_tapisExecSystemOutputDir:?Tapis output directory is not set}"
WORKFLOW="${SSM_WORKFLOW:-dictionary}"
mkdir -p "$OUT" "${XDG_CACHE_HOME:-/tmp/ssm-cache}"

args=("$WORKFLOW" --output-dir "$OUT")
if [ -f "$IN/job.yaml" ]; then args+=(--config "$IN/job.yaml"); fi
if [ -e "$IN/corpus" ]; then args+=(--corpus "$IN/corpus"); fi
if [ -f "$IN/lexicon.json" ]; then
  args+=(--lexicon "$IN/lexicon.json")
elif [ -f "$IN/lexicon.csv" ]; then
  args+=(--lexicon "$IN/lexicon.csv")
fi
if [ -f "$IN/taxonomy.json" ]; then args+=(--taxonomy "$IN/taxonomy.json"); fi

semantic-sensemaking --version
echo "Running: semantic-sensemaking ${args[*]}"
exec semantic-sensemaking "${args[@]}"
