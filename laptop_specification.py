class Specification():
    def __init__(self):
        self.manufacture=[]

    def manufacturer(self,df):
        manufact = df.groupby(['Manufacturer'])['Model Name'].apply(list)
        brand = {}
        for model in manufact.index:
            model_name = list(set([x.lower()  for x in manufact[model]]))
            brand[model]=model_name
        company = list(brand.keys())
        return company

    def category(self,df):
        category_list = list(df['Category'].unique())
        return category_list

    def screen_size(self,df):
        Inches = list(df['Screen Size'].value_counts().index)
        return Inches


    def cpu(self,df):
        cpu_list = list(df['CPU'].value_counts().index)
        return cpu_list

    def get_ram(self,df):
        RAM = list(df['RAM'].value_counts().index)
        return RAM

    def gpu(self,df):
        gpu_list = list(df['GPU'].value_counts().index)
        return gpu_list

    def operating_system(self,df):
        os = list(df['Operating System'].value_counts().index)
        return os

    def os_ver(self, df):
        os_ver = list(df['Operating System Version'].value_counts().index)
        return os_ver

    def resolution(self, df):
        res = list(df['resolution'].value_counts().index)
        return res

    def screen_type(self, df):
        sc_type= list(df['screentype'].value_counts().index)
        return sc_type

    def weight(self, df):
        w = list(df['Weight'].value_counts().index)
        return w

    def touch_screen(self,df):
        l = list(df['touchscreen'].value_counts().index)
        return l



