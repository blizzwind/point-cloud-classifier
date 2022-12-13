import laspy
import numpy as np
import pandas as pd
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def grounded_outline(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(100, activation="relu"),
      layers.Dense(100, activation="relu"),
      layers.Dense(2)
  ])
  model.compile(loss="mean_absolute_error",optimizer=tf.keras.optimizers.Adam(0.001))
  return model

try:
  shutil.rmtree("ckpt")
except:
  pass
normalizer = tf.keras.layers.Normalization(axis=-1)
grounded = grounded_outline(normalizer)
feature_df = pd.DataFrame(columns=["voxelmeanz","voxel1minz","voxel1maxz","voxel1meanz","voxel20minz","voxel20maxz","voxel20meanz","voxel40minz","voxel40maxz","voxel40meanz"])
label_df = pd.DataFrame(columns=["voxelclass","class"])
file = []
for i in os.listdir("input/train"):
    file.append(i)
for i in file:
    las = laspy.read("input/train/"+i)
    df = pd.DataFrame(columns=["x","y","z","class"])
    df["x"] = np.array(las.x)
    df["y"] = np.array(las.y)
    df["z"] = np.array(las.z)
    df["minx"] = df["x"] - min(df["x"])
    df["miny"] = df["y"] - min(df["y"])
    df["minz"] = df["z"] - min(df["z"])
    df["class"] = np.array(las.classification)
    df["voxelx"] = df["minx"] // 0.2
    df["voxely"] = df["miny"] // 0.2
    df["voxelz"] = df["minz"] // 0.2
    tmp = df.groupby(["voxelx","voxely","voxelz"]).agg({"class":"mean"}).reset_index()
    tmp["class"][tmp["class"] == 31] = 0
    tmp["class"][tmp["class"] > 0] = 1
    tmp["voxelclass"] = tmp["class"]
    tmp = tmp.drop(columns=["class"])
    df = df.merge(tmp,on=["voxelx","voxely","voxelz"])
    df["class"] = df["voxelclass"]
    df["class"][df["class"] == 1] = 2
    df["class"][df["class"] == 0] = 1
    df["class"][df["class"] == 2] = 0
    df["voxelmeanzz"] = df["minz"]
    tmp = df.groupby(["voxelx","voxely","voxelz"]).agg({"voxelmeanzz":"mean"}).reset_index()
    tmp["voxelmeanz"] = tmp["voxelmeanzz"]
    tmp = tmp.drop(columns=["voxelmeanzz"])
    df = df.merge(tmp,on=["voxelx","voxely","voxelz"])
    df = df.drop(columns=["voxelmeanzz"])
    df = df.drop(columns=["voxelx","voxely","voxelz","x","y","z"])
    df["voxelx1"] = df["minx"] // 1
    df["voxely1"] = df["miny"] // 1
    df["voxelz1"] = df["minz"] // 1
    df["voxel1minzz"] = df["minz"]
    tmp = df.groupby(["voxelx1","voxely1","voxelz1"]).agg({"voxel1minzz":"min"}).reset_index()
    tmp["voxel1minz"] = tmp["voxel1minzz"]
    tmp = tmp.drop(columns=["voxel1minzz"])
    df = df.merge(tmp,on=["voxelx1","voxely1","voxelz1"])
    df = df.drop(columns=["voxel1minzz"])
    df["voxel1maxzz"] = df["minz"]
    tmp = df.groupby(["voxelx1","voxely1","voxelz1"]).agg({"voxel1maxzz":"max"}).reset_index()
    tmp["voxel1maxz"] = tmp["voxel1maxzz"]
    tmp = tmp.drop(columns=["voxel1maxzz"])
    df = df.merge(tmp,on=["voxelx1","voxely1","voxelz1"])
    df = df.drop(columns=["voxel1maxzz"])
    df["voxel1meanzz"] = df["minz"]
    tmp = df.groupby(["voxelx1","voxely1","voxelz1"]).agg({"voxel1meanzz":"mean"}).reset_index()
    tmp["voxel1meanz"] = tmp["voxel1meanzz"]
    tmp = tmp.drop(columns=["voxel1meanzz"])
    df = df.merge(tmp,on=["voxelx1","voxely1","voxelz1"])
    df = df.drop(columns=["voxel1meanzz"])
    df = df.drop(columns=["voxelx1","voxely1","voxelz1"])
    df["voxelx20"] = df["minx"] // 20
    df["voxely20"] = df["miny"] // 20
    df["voxelz20"] = df["minz"] // 20
    df["voxel20minzz"] = df["minz"]
    tmp = df.groupby(["voxelx20","voxely20","voxelz20"]).agg({"voxel20minzz":"min"}).reset_index()
    tmp["voxel20minz"] = tmp["voxel20minzz"]
    tmp = tmp.drop(columns=["voxel20minzz"])
    df = df.merge(tmp,on=["voxelx20","voxely20","voxelz20"])
    df = df.drop(columns=["voxel20minzz"])
    df["voxel20maxzz"] = df["minz"]
    tmp = df.groupby(["voxelx20","voxely20","voxelz20"]).agg({"voxel20maxzz":"max"}).reset_index()
    tmp["voxel20maxz"] = tmp["voxel20maxzz"]
    tmp = tmp.drop(columns=["voxel20maxzz"])
    df = df.merge(tmp,on=["voxelx20","voxely20","voxelz20"])
    df = df.drop(columns=["voxel20maxzz"])
    df["voxel20meanzz"] = df["minz"]
    tmp = df.groupby(["voxelx20","voxely20","voxelz20"]).agg({"voxel20meanzz":"mean"}).reset_index()
    tmp["voxel20meanz"] = tmp["voxel20meanzz"]
    tmp = tmp.drop(columns=["voxel20meanzz"])
    df = df.merge(tmp,on=["voxelx20","voxely20","voxelz20"])
    df = df.drop(columns=["voxel20meanzz"])
    df = df.drop(columns=["voxelx20","voxely20","voxelz20"])
    df["voxelx40"] = df["minx"] // 40
    df["voxely40"] = df["miny"] // 40
    df["voxelz40"] = df["minz"] // 40
    df["voxel40minzz"] = df["minz"]
    tmp = df.groupby(["voxelx40","voxely40","voxelz40"]).agg({"voxel40minzz":"min"}).reset_index()
    tmp["voxel40minz"] = tmp["voxel40minzz"]
    tmp = tmp.drop(columns=["voxel40minzz"])
    df = df.merge(tmp,on=["voxelx40","voxely40","voxelz40"])
    df = df.drop(columns=["voxel40minzz"])
    df["voxel40maxzz"] = df["minz"]
    tmp = df.groupby(["voxelx40","voxely40","voxelz40"]).agg({"voxel40maxzz":"max"}).reset_index()
    tmp["voxel40maxz"] = tmp["voxel40maxzz"]
    tmp = tmp.drop(columns=["voxel40maxzz"])
    df = df.merge(tmp,on=["voxelx40","voxely40","voxelz40"])
    df = df.drop(columns=["voxel40maxzz"])
    df["voxel40meanzz"] = df["minz"]
    tmp = df.groupby(["voxelx40","voxely40","voxelz40"]).agg({"voxel40meanzz":"mean"}).reset_index()
    tmp["voxel40meanz"] = tmp["voxel40meanzz"]
    tmp = tmp.drop(columns=["voxel40meanzz"])
    df = df.merge(tmp,on=["voxelx40","voxely40","voxelz40"])
    df = df.drop(columns=["voxel40meanzz"])
    df = df.drop(columns=["voxelx40","voxely40","voxelz40"])
    feature_df = pd.concat([feature_df,df],join="inner",ignore_index=True)
    label_df = pd.concat([label_df,df],join="inner",ignore_index=True)
grounded.fit(feature_df,label_df,epochs=100,batch_size=102400)
grounded.save_weights("ckpt/ckpt")
