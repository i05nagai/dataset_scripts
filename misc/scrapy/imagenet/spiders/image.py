# -*- coding: utf-8 -*-
import scrapy


class ImageSpider(scrapy.Spider):
    name = 'image'
    allowed_domains = ['image-net.org']

    def __init__(self, wnid='n00021265'):
        super().__init__()
        url = self._get_url_for_hyponym(wnid)
        self.wnid = wnid
        self.start_urls = [
            url,
        ]

    # parse url in start_urls
    def parse(self, response):
        wnids, urls = self._parse_hyponym(response.body)
        print(' hyponym: {0}'.format(wnids))
        for wnid, url in zip(wnids, urls):
            yield scrapy.Request(
                url,
                callback=self.parse_item,
                meta={'wnid': wnid})

    def parse_item(self, response):
        urls = self._parse_image_urls(response.body)
        print(' urls:{0}'.format(urls))
        wnid = response.meta['wnid']
        yield {
            'file_urls': urls,
            'wnid': wnid,
        }

    def _parse_hyponym(self, data, include_parent=True):
        data = data.decode('utf8').split('\r\n')
        if include_parent:
            wnids = data[0:-1]
        else:
            wnids = data[1:-1]
        wnids = [hyponym.replace(r'-', '') for hyponym in wnids]
        return (wnids, [self._get_url_for_image_urls(hyponym) for hyponym in wnids])

    def _parse_image_urls(self, data, max_urls=None):
        data = data.decode('utf-8').split('\r\n')
        if max_urls is None:
            urls = data[0:-1]
        else:
            urls = data[0:max_urls]
        return urls

    def _get_url_for_image_urls(self, wnid):
        baseurl = 'http://www.image-net.org/api'
        url = '{0}/text/imagenet.synset.geturls?wnid={1}'
        return url.format(baseurl, wnid)

    def _get_url_for_hyponym(self, wnid):
        baseurl = 'http://www.image-net.org/api'
        url = '{0}/text/wordnet.structure.hyponym?wnid={1}&full=1'
        return url.format(baseurl, wnid)
