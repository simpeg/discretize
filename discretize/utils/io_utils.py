"""Simple input/output routines."""
from urllib.request import urlretrieve
import os
import importlib
import json


def load_mesh(file_name):
    """Load discretize mesh saved to json file.

    For a discretize mesh that has been converted to dictionary and
    written to a json file, the function **load_mesh** loads the
    json file and reconstructs the mesh object.

    Parameters
    ----------
    file_name : str
        Name of the json file being read in. Contains all information required to
        reconstruct the mesh.

    Returns
    -------
    discretize.base.BaseMesh
        A discretize mesh defined by the class and parameters stored in the json file
    """
    with open(file_name, "r") as outfile:
        jsondict = json.load(outfile)
        module_name = jsondict.pop(
            "__module__", "discretize"
        )  # default to loading from discretize
        class_name = jsondict.pop("__class__")
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        if "_n" in jsondict:
            jsondict["shape_cells"] = jsondict.pop(
                "_n"
            )  # need to catch this old _n property here
        data = cls(**jsondict)
    return data


def download(url, folder=".", overwrite=False, verbose=True):
    """
    Download file(s) stored in a cloud directory.

    Parameters
    ----------
    url : str or list of str
        url or list of urls for the file(s) being downloaded
    folder : str, optional
        Local folder where downloaded files are to be stored
    overwrite : bool, optional
        Overwrite files if they have the same name as newly downloaded files
    verbose : bool, optional
        Print progress when downloading multiple files

    Returns
    -------
    os.path or list of os.path
        The path or a list of paths for all downloaded files
    """

    def rename_path(downloadpath):
        splitfullpath = downloadpath.split(os.path.sep)

        # grab just the file name
        fname = splitfullpath[-1]
        fnamesplit = fname.split(".")
        newname = fnamesplit[0]

        # check if we have already re-numbered
        newnamesplit = newname.split("(")

        # add (num) to the end of the file name
        if len(newnamesplit) == 1:
            num = 1
        else:
            num = int(newnamesplit[-1][:-1])
            num += 1

        newname = "{}({}).{}".format(newnamesplit[0], num, fnamesplit[-1])
        return os.path.sep.join(splitfullpath[:-1] + newnamesplit[:-1] + [newname])

    # ensure we are working with absolute paths and home directories dealt with
    folder = os.path.abspath(os.path.expanduser(folder))

    # make the directory if it doesn't currently exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(url, str):
        file_names = [url.split("/")[-1]]
    elif isinstance(url, list):
        file_names = [u.split("/")[-1] for u in url]

    downloadpath = [os.path.sep.join([folder, f]) for f in file_names]

    # check if the directory already exists
    for i, download in enumerate(downloadpath):
        if os.path.exists(download):
            if overwrite:
                if verbose:
                    print("overwriting {}".format(download))
            else:
                while os.path.exists(download):
                    download = rename_path(download)

                if verbose:
                    print("file already exists, new file is called {}".format(download))
                downloadpath[i] = download

    # download files
    urllist = url if isinstance(url, list) else [url]
    for u, f in zip(urllist, downloadpath):
        print("Downloading {}".format(u))
        urlretrieve(u, f)
        print("   saved to: " + f)

    print("Download completed!")
    return downloadpath if isinstance(url, list) else downloadpath[0]
