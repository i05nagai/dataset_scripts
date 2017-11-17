# -*- coding: utf-8 -*-
import scrapy


class GoogleimageSpider(scrapy.Spider):
    name = 'googleimage'
    allowed_domains = [
        'www.google.co.jp',
        'www.google.com',
        'images.google.com',
    ]

    def __init__(self, search_word='bread'):
        super().__init__()

        url = 'https://www.google.co.jp/search?'
        url = 'https://images.google.com/'
        self.start_urls = [
            url,
        ]
        self.search_word = search_word
        self.save_dir = ''

    def parse(self, response):
        yield scrapy.FormRequest.from_response(
            response,
            formdata={'lst-ib': self.search_word},
            callback=self.parse_after_search
        )

    def parse_after_search(self, response):
        urls = []
        # print(response.css('img.rg_ic.rg_i'))
        print(response.css('#ires'))
        yield {
            'file_urls': urls,
        }
