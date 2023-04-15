import os
import argparse
import matplotlib.pyplot as plt

from pathlib import Path

from rowrectifier.rowrectifier import RowRectifier


def rectify(inpath, outpath):

    # Determine files to process
    files_to_rectify = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    if os.path.isfile(inpath):
        files_to_rectify = [Path(inpath)]
    elif os.path.isdir(inpath):
        files_to_rectify = [f for f in Path(inpath).glob('*')
                            if f.suffix.lower() in image_extensions]
        # for file_path in Path(inpath).glob("*"):  # or **/*
        #     if file_path.is_file():
        #         files_to_rectify.append(file_path)
    else:
        return f"{inpath} does not exist."

    # Process
    for fpath in files_to_rectify:

        # Load image
        Img = plt.imread(fpath)

        # Rectify
        rr = RowRectifier(method='univariate_spline', 
                          s=None, n_poly_sides=3)
        rr.fit(Img)
        # Iout, Isrc_polys, Isrc_ridges, Idst_polys, Idst_ridges = \
        #     rr.transform(Img, return_intermediates=True)
        Iout = rr.transform(Img)

        outfilename = Path(fpath).stem + '_out' + Path(fpath).suffix
        if outpath is not None:
            outfilename = Path.joinpath(Path(outpath), Path(outfilename))

        plt.imsave(outfilename, Iout)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Rectify a file or a folder.")
    parser.add_argument("inpath", type=str, help="Path to check. Can be a file or a folder of files.")
    parser.add_argument("--outfolder", type=str, help="Output folder", default=None)
    args = parser.parse_args()

    rectify(args.inpath, args.outfolder)
