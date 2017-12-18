import os


PATH_TO_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PATH_TO_TEMPLATE_HTML = os.path.join(PATH_TO_THIS_DIR, 'template.html')
PATH_TO_TEMPLATE_CSS = os.path.join(PATH_TO_THIS_DIR, 'template.css')


def _read_template(path):
    try:
        with open(path, 'r') as f:
            data = f.read()
    except IOError as e:
        print(e)
    return data


def _gen_html(body):
    str_css = _read_template(PATH_TO_TEMPLATE_CSS)
    str_html = _read_template(PATH_TO_TEMPLATE_HTML)
    str_html = str_html.replace('@TEMPLATE_CSS', str_css)
    str_html = str_html.replace('@TEMPLATE_BODY', body)
    return str_html


def _make_image_gallery_html(images):
    html_template = """
    <div class="flex-container">
        {0}
    </div>
    """
    item_template = """
      <div class="flex-item">
        <img  class="flex-item__image" src="{0}"/>
      </div>
    """
    item_str = ''
    for image in images:
        item_str += item_template.format(image)
    html = html_template.format(item_str)
    return html


def make_image_gallery(images_list):
    body = ''
    for images in images_list:
        body += _make_image_gallery_html(images)
    return _gen_html(body)


def save_image_gallery(images_list, path_to_output):
    """save_image_gallery

    :param images_list:
    :param path_to_output:
    """
    html = make_image_gallery(images_list)
    try:
        with open(path_to_output, 'w') as f:
            f.write(html)
    except IOError as e:
        print(e)
