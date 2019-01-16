import subprocess
import unittest
import os


class Doc_Test(unittest.TestCase):

    @property
    def path_to_docs(self):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        return dirname.split(os.path.sep)[:-2] + ['docs']

    @property
    def path_to_build(self):
        return self.path_to_docs + ["_build"]

    @property
    def path_to_doctrees(self):
        return self.path_to_build + ["doctrees"]

    @property
    def path_to_html(self):
        return self.path_to_build + ["html"]

    @property
    def path_to_api(self):
        return self.path_to_build + ["api"]

    def setUp(self):

        subprocess.call([
           "sphinx-autogen", "-i", "-t", "_templates", "-o",
           "{}".format(os.path.sep.join(self.path_to_api + ["generated"])),
           "{}".format(os.path.sep.join(self.path_to_api + ["index.rst"]))
        ])

    def test_html(self):

        check = subprocess.call([
            "sphinx-build",
            # "-nW",
            "-b", "html", "-d",
            "{0!s}".format((os.path.sep.join(self.path_to_doctrees))),
            "{0!s}".format((os.path.sep.join(self.path_to_docs))),
            "{0!s}".format((os.path.sep.join(self.path_to_html)))
        ])
        assert check == 0

    # def test_latex(self):
    #     path_to_doctrees = os.path.sep.join(self.path_to_docs.split(os.path.sep) + ['_build']+['doctrees'])
    #     latex_path = os.path.sep.join(self.path_to_docs.split(os.path.sep) + ['_build']+['latex'])

    #     check = subprocess.call(["sphinx-build", "-nW", "-b", "latex", "-d",
    #         "%s"%(path_to_doctrees),
    #         "%s"%(self.path_to_docs),
    #         "%s"%(latex_path)])
    #     assert check == 0

    def test_linkcheck(self):
        link_path = os.path.sep.join(self.path_to_docs + ['_build'])

        check = subprocess.call([
            "sphinx-build",
            # "-nW",
            "-b", "linkcheck", "-d",
            "%s"%(os.path.sep.join(self.path_to_doctrees)),
            "%s"%(os.path.sep.join(self.path_to_docs)),
            "%s"%(os.path.sep.join(self.path_to_build + ["linkcheck"]))
        ])
        assert check == 0

if __name__ == '__main__':
    unittest.main()
