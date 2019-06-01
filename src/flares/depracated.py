import matplotlib.cbook as cbook
import matplotlib.image as image


# Apply watermark
def apply_watermark():
    image_file = os.path.abspath('greyscale_logo_no_text_transparent_bg.png')
    datafile = cbook.get_sample_data(image_file, asfileobj=False)
    watermark = image.imread(datafile)
    # watermark[:, :, -1] = 0.5  # set the alpha channel

    # for plots_with_data in plots_with_data_by_factory.values():
    #    for plot in plots_with_data.plots:
    #        plot.figure.figimage(watermark, 10, 10, resize=True, zorder=3)

def watermark_plot(plot):
    # TODO: change to diagonal 'Haifa-ERC' text
    x_offset = int((plot.figure.bbox.xmax * 0.5 - watermark.shape[0]/2))
    y_offset = int((plot.figure.bbox.ymax * 0.5 - watermark.shape[1]/2))
    plot.figure.figimage(watermark, x_offset, y_offset, zorder=1, alpha=0.04)
