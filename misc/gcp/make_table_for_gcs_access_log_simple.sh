#!/bin/bash

################################################################################
# Requirements:
#  * gsutil
#  * curl
# See:
#  * https://cloud.google.com/storage/docs/access-logs
################################################################################

usage() {
  cat <<EOF
make_table.sh is a tool for make tables from access logs of a bucket

Usage:
    make_table.sh <bigquery_dataset>
EOF
}

################################################################################
# Description:
# Globals:
#  DATASET
# Arguments:
#   uri: path to object in the bucket
#   prefix: "storage" or "usage"
# Returns:
#   None
################################################################################
load_data_to_table() {
  local uri=$1
  local prefix=$2
  bq load --skip_leading_rows=1 \
    ${DATASET}.${prefix} \
    ${uri}_${prefix}_* \
    "./cloud_storage_${prefix}_schema_v0.json"
}

# Constants
declare -r BUCKET_NAME_FOR_LOG="WRITE_SOME_BUCKET_NAME"
declare -r SCHEMA_FOR_USAGE_LOG="http://storage.googleapis.com/pub/cloud_storage_usage_schema_v0.json"
declare -r SCHEMA_FOR_STORAGE_LOG="http://storage.googleapis.com/pub/cloud_storage_storage_schema_v0.json"
# Arguments
declare -r DATASET=$1
if [ -z ${DATASET+x} ]
then
  usage
  exit 0
fi

# download schema
curl -L -O ${SCHEMA_FOR_USAGE_LOG}
curl -L -O ${SCHEMA_FOR_STORAGE_LOG}

bq mk ${DATASET}.usage
bq mk ${DATASET}.storage
# make tables
for uri in $(gsutil ls "gs://${BUCKET_NAME_FOR_LOG}"); do
  bucket_name=$(basename $uri)

  load_data_to_table ${uri} "usage"
  load_data_to_table ${uri} "storage"
done

