# -*- coding: utf-8 -*-
import scrapy.pipelines.files
import os
import errno

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


def make_directory(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class ImagenetPipeline(scrapy.pipelines.files.FilesPipeline):

    def __init__(self, store_uri, download_func=None, settings=None):
        super().__init__(store_uri, download_func, settings)

    def get_media_requests(self, item, info):
        if 'wnid' in item:
            urls = item.get(self.files_urls_field, [])
            meta = {
                'wnid': item['wnid'],
            }
            return [scrapy.Request(x, meta=meta) for x in urls]
        else:
            return super().get_media_requests(item, info)

    def file_path(self, request, response=None, info=None):
        path = super().file_path(request, response, info)
        if 'wnid' in request.meta:
            wnid = request.meta['wnid']
            dirs = path.split('/')
            path = os.path.join(dirs[0], wnid, dirs[-1])
            return path
        else:
            return path


class GoogleImagePipeline(scrapy.pipelines.files.FilesPipeline):

    def __init__(self, store_uri, download_func=None, settings=None):
        super().__init__(store_uri, download_func, settings)

    def get_media_requests(self, item, info):
        if 'wnid' in item:
            urls = item.get(self.files_urls_field, [])
            meta = {
                'wnid': item['wnid'],
            }
            return [scrapy.Request(x, meta=meta) for x in urls]
        else:
            return super().get_media_requests(item, info)

    def file_path(self, request, response=None, info=None):
        path = super().file_path(request, response, info)
        if 'wnid' in request.meta:
            wnid = request.meta['wnid']
            dirs = path.split('/')
            path = os.path.join(dirs[0], wnid, dirs[-1])
            return path
        # 
        else:
            dirs = path.split('/')
            path = os.path.join(dirs[0], 'google', dirs[-1])
            return path
