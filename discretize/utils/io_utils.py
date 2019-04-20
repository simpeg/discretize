import urllib
import os
import sys

def download(
    url, folder='.', overwrite=False, verbose=True
):
    """
    Function to download all files stored in a cloud directory

    :param str url: url or list of urls for the file(s) to be downloaded
    :param str folder: folder to where the directory is created and files downloaded (default is the current directory)
    :param bool overwrite: overwrite if a file with the specified name already exists
    :param bool verbose: print out progress
    """

    def rename_path(downloadpath):
        splitfullpath = downloadpath.split(os.path.sep)

        # grab just the filename
        fname = splitfullpath[-1]
        fnamesplit = fname.split('.')
        newname = fnamesplit[0]

        # check if we have already re-numbered
        newnamesplit = newname.split('(')

        # add (num) to the end of the filename
        if len(newnamesplit) == 1:
            num = 1
        else:
            num = int(newnamesplit[-1][:-1])
            num += 1

        newname = '{}({}).{}'.format(newnamesplit[0], num, fnamesplit[-1])
        return os.path.sep.join(
            splitfullpath[:-1] + newnamesplit[:-1] + [newname]
        )

    # grab the correct url retriever
    if sys.version_info < (3,):
        urlretrieve = urllib.urlretrieve
    else:
        urlretrieve = urllib.request.urlretrieve

    # ensure we are working with absolute paths and home directories dealt with
    folder = os.path.abspath(os.path.expanduser(folder))

    # make the directory if it doesn't currently exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    if isinstance(url, str):
        filenames = [url.split('/')[-1]]
    elif isinstance(url, list):
        filenames = [u.split('/')[-1] for u in url]

    downloadpath = [os.path.sep.join([folder, f]) for f in filenames]

    # check if the directory already exists
    for i, download in enumerate(downloadpath):
        if os.path.exists(download):
            if overwrite is True:
                if verbose is True:
                    print("overwriting {}".format(download))
            elif overwrite is False:
                while os.path.exists is True:
                    download = rename_path(download)

                if verbose is True:
                    print(
                        "file already exists, new file is called {}".format(
                            download
                        )
                    )
                downloadpath[i] = download

    # download files
    urllist = url if isinstance(url, list) else [url]
    for u, f in zip(urllist, downloadpath):
        print("Downloading {}".format(u))
        urlretrieve(u, f)
        print("   saved to: " + f)

    print("Download completed!")
    return downloadpath if isinstance(url, list) else downloadpath[0]
