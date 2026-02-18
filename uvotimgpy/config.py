# project_paths.py
from pathlib import Path

class ProjectPaths:
    def __init__(self):
        self.work = Path("/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork")
        #self.data = self.work / "data"
        #self.projects = self.work / "projects"
        #self.packages = self.work / "packages"
        self.data = self.work.joinpath('data')
        self.projects = self.work.joinpath('projects')
        self.packages = self.work.joinpath('packages')
    
    def get_subpath(self, base_path, *args):
        if isinstance(base_path, str):
            base_path = Path(base_path)
        return base_path.joinpath(*args)
    
    @property
    def project_29p_hst(self):
        return self.get_subpath(self.projects, "29p", "HST")
    
    @property
    def package_uvotimgpy(self):
        return self.get_subpath(self.packages, "uvotimgpy")

paths = ProjectPaths()  # 创建单例实例