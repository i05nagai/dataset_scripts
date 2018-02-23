# coding: utf-8
import MySQLdb
from ..util import json_helper


def get_connection(path):
    credential = json_helper.read_json(path)
    conn = MySQLdb.connect(
        user=credential['username'],
        passwd=credential['password'],
        host=credential['host'],
        port=credential['port'],
        db=credential['db'],
        charset='charset')
    return conn


def run_query_from_path(conn, path_to_query):
    try:
        with open(path_to_query, 'r') as f:
            query = f.readlines()
    except IOError as e:
        print(e)
    return run_query(conn, query)


def run_query(conn, query):
    c = conn.cursor()
    c.execute(query)

    data = [row for row in c.fetchall()]
    c.close()
    return data
