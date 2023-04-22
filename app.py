from flask import Flask, render_template, request
from  laptop_specification import Specification
from model_train import Prediction
import pickle as pkl

# importing data from pickle file
with open("clean_df.pkl", "rb") as f:
    df = pkl.load(f)
columns = df.columns

# creating object of specification class to access all specification
storage = list(df[' Storage'].value_counts().index)
laptop = Specification()
manufacturer = laptop.manufacturer(df)
model_name = laptop.manufacturer(df)
category = laptop.category(df)
screen_size = laptop.screen_size(df)
cpu = laptop.cpu(df)
ram = laptop.get_ram(df)
gpu = laptop.gpu(df)
os = laptop.operating_system(df)
os_ver = laptop.os_ver(df)
resolution = laptop.resolution(df)
screen_type = laptop.screen_type(df)
weight = laptop.weight(df)
touchscreen = [False,True]
screen_type = screen_type[1:]
print(screen_type)
app = Flask(__name__)
app.static_folder = 'static'

print(df.columns)
@app.route('/')
def index():
    return render_template('home.html', manufacturers=manufacturer, model_names=model_name, category=category , screen_size=screen_size, cpu=cpu , ram=ram, storage=storage, gpu=gpu, os=os, os_ver=os_ver , weight=weight, resolution=resolution , touchscreen=touchscreen, screen_type=screen_type )


@app.route('/submit', methods=['POST'])
def submit():
    data_f ={}
    for col in columns:
        l = []
        if col=='touchscreen':
            if request.form.get(col)==True:
                l.append(1.0)
            else:
                l.append(0.0)
        else:
            l.append(request.form.get(col))
        data_f[col] = l
    # d = pd.DataFrame(data_f)
    # data_prep = Data_preparation()
    # clean_data = data_prep.clean(d)
    obj_price = Prediction()
    price = obj_price.predict_answer(data_f)

    return render_template('submit.html', predicted_price=price)


if __name__ == '__main__':
    app.run(debug=True)
