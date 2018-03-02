import base64
import io

from scipy import misc


def decode(image):
  buff = io.BytesIO()

  misc.toimage(image[:, :, 0]).save(buff, format='png')

  return 'data:image/png;base64,' + base64.b64encode(buff.getvalue()).decode()


class ImageWriter:
  """Creates a file and writes images to HTML table."""

  def __init__(self, filename):
    """Creates an ImageWriter.

    Args:
      filename: name of file to create or overwrite
      labels: list of strings to use as table headers
    Returns
      writer: ImageWriter instance
    """
    self.filename = filename

    self._body = False
    self._file = None

  def __enter__(self):
    self._file = open(self.filename, 'w+')
    self._file.write('<html><body><table>')

    return self

  def __exit__(self, *args):
    self._file.write('</tbody></table></body></html>')
    self._file.close()

  def write_head(self, labels):
    """Appends file with row of images.

    Args:
      images: list of numpy images
    """
    if self._body:
      raise RuntimeError('cannot write head after rows')

    self._file.write('<thead><tr>')

    for l in labels:
      self._file.write(f'<th>{l}</th>')

    self._file.write('</tr></thead>')

  def write_row(self, images):
    """Appends file with row of images.

    Args:
      images: list of numpy images
    """
    if not self._body:
      self._file.write('<tbody>')
      self._body = True

    self._file.write('<tr>')

    for image in images:
      self._file.write(f'<td><img src="{decode(image)}" /></td>')

    self._file.write('</tr>')
