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
make_table_for_gcs_access_log.sh is a tool for make tables from access logs of a bucket

Usage:
    make_table.sh <bucket_name> <bigquery_dataset>
EOF
}

################################################################################
# Description
# Arguments:
#   object_name: such as 
#     "_usage_2018_01_10_08_00_00_0d5dd_v0"
#     "_storage_2018_01_10_08_00_00_0d5dd_v0"
#   prefix: "storage" or "usage"
# Returns:
#   "${prefix}_yyyymmdd"
################################################################################
get_table_name()
{
  local object_name=$1
  local prefix=$2
  echo ${object_name} | sed "s/_${prefix}_\([0-9]\{4\}\)_\([0-9][0-9]\)_\([0-9][0-9]\)_.*/${prefix}_\1\2\3/"
}

################################################################################
# Description
# Globals:
#  BUCKET_NAME_FOR_LOG
#  DATASET
# Arguments:
#   object_name: such as 
#     "_usage_2018_01_10_08_00_00_0d5dd_v0"
#     "_storage_2018_01_10_08_00_00_0d5dd_v0"
#   prefix: "storage" or "usage"
# Returns:
#   "${prefix}_yyyymmdd"
################################################################################
make_tables() {
  local BUCKET_NAME=$1

  for uri in $(gsutil ls "gs://${BUCKET_NAME_FOR_LOG}/${BUCKET_NAME}"); do
    object_name=$(basename $uri)
    # get prefix
    if [ `echo ${object_name} | grep 'storage'` ]
    then
      prefix=storage
    elif [ `echo ${object_name} | grep 'usage'` ]
    then
      prefix=usage
    fi
    table_name=$(get_table_name ${object_name} ${prefix})

    echo "  Making ${DATASET}.${table_name} ..."
    bq load --skip_leading_rows=1 \
      ${DATASET}.${table_name} \
      ${uri} \
      "./cloud_storage_${prefix}_schema_v0.json"
  done
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


# make tables
for uri in $(gsutil ls "gs://${BUCKET_NAME_FOR_LOG}"); do
  bucket_name=$(basename $uri)
  make_tables ${BUCKET_NAME_FOR_LOG} ${bucket_name} ${DATASET}
done
