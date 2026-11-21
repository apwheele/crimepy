"""Plot helpers."""

from importlib.resources import as_file, files

import matplotlib.pyplot as plt

import crimepy.cdcplot as cdcplot


def test_add_logo_from_path_string():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with as_file(files("crimepy").joinpath("CDCWLineRec.PNG")) as logo_path:
        cdcplot.add_logo(ax, logo=str(logo_path))
    images = [img for inset in ax.child_axes for img in inset.get_images()]
    assert images
    assert images[0].get_array().size > 0
    plt.close(fig)
