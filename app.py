""" streamlit_demo
streamlitでIrisデータセットの分析結果を Web アプリ化するモジュール

【Streamlit 入門 1】Streamlit で機械学習のデモアプリ作成 – DogsCox's tech. blog
https://dogscox-trivial-tech-blog.com/posts/streamlit_demo_iris_decisiontree/
"""

from itertools import chain
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 
# import graphviz
# import plotly.graph_objects as go
# irisデータセットでテストする
# from sklearn.datasets import load_iris

# 決定木で分類してみる
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# ロゴの表示用
from PIL import Image


sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def st_display_table(df: pd.DataFrame):

    # データフレームを表示
    st.subheader('データの確認')
    # st.caption('最初の10件のみ表示しています')
    st.table(df)

    # Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def st_display_histogram(df: pd.DataFrame, x_col, hue_col):

    fig, ax = plt.subplots()
    # plt.title("ヒストグラム", fontsize=20)     # (3) タイトル
    # plt.xlabel("Age", fontsize=20)          # (4) x軸ラベル
    # plt.ylabel("Frequency", fontsize=20)      # (5) y軸ラベル
    plt.grid(True)                            # (6) 目盛線の表

    if hue_col == 'null':
        unique_cnt = len(df[x_col].value_counts())
        if unique_cnt > 10:
            plt.xlabel(x_col, fontsize=12)          # x軸ラベル
            plt.hist(df[x_col])   # 単なるヒストグラム
        else:
            # print(len(df[x_col].value_counts()))
            sns.countplot(data=df, x=x_col, ax=ax)
    else:
        sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax)

    st.pyplot(fig)

    # seabornでグラフを複数のグラフを描画する - Qiita
    # https://qiita.com/tomokitamaki/items/b954e26be739bee5621e



def ml_dtree_pred(
    X: pd.DataFrame,
    y: pd.Series,
    depth: int,
    t_size: float) -> list:
    """ 決定木で学習、予測を行う関数
    Irisデータセット全体で学習し、学習データの予測値を返す関数
    Args:
        X(pd.DataFrame): 説明変数郡
        y(pd.Series): 目的変数
    
    Returns:
        List: [モデル, 学習データを予測した予測値, accuracy]のリスト
    """

    # train_test_split関数を利用してデータを分割する
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, train_size=t_size, random_state=0, stratify=y)

    # 学習
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(train_x, train_y)

    # 訓練データで予測 ＆ 精度評価
    train_pred = clf.predict(train_x)
    train_score = accuracy_score(train_y, train_pred)

    # 訓練データで予測 ＆ 精度評価
    valid_pred = clf.predict(valid_x)
    valid_score = accuracy_score(valid_y, valid_pred)

    return [clf, train_pred, train_score, valid_pred, valid_score]


def ml_dtree(
    X: pd.DataFrame,
    y: pd.Series,
    depth: int) -> list:
    """ 決定木で学習、予測を行う関数
    Irisデータセット全体で学習し、学習データの予測値を返す関数
    Args:
        X(pd.DataFrame): 説明変数郡
        y(pd.Series): 目的変数
    
    Returns:
        List: [モデル, 学習データを予測した予測値, accuracy]のリスト
    """
    # 学習
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X, y)

    # 予測
    pred = clf.predict(X)

    # accuracyで精度評価
    score = accuracy_score(y, pred)

    return [clf, pred, score]


def st_display_dtree(clf, features):
    """決定木可視化関数
    streamlitでDtreeVizによる決定木を可視化する関数
    Args:
        clf(sklearn.DecisionTreeClassifier): 学習済みモデル
    Return:
    """
    # # graphvizで決定木を可視化
    # dot = tree.export_graphviz(clf, out_file=None)
    # # stで表示する
    # st.graphviz_chart(dot)

    dot = tree.export_graphviz(clf, 
                               out_file=None, # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                               filled=True, # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                               rounded=True, # Trueにすると、ノードの角を丸く描画する。
                            #    feature_names=['あ', 'い', 'う', 'え'], # これを指定しないとチャート上で特徴量の名前が表示されない
                               feature_names=features, # これを指定しないとチャート上で特徴量の名前が表示されない
                            #    class_names=['setosa' 'versicolor' 'virginica'], # これを指定しないとチャート上で分類名が表示されない
                               special_characters=True # 特殊文字を扱えるようにする
                               )

    # stで表示する
    st.graphviz_chart(dot)


def st_display_rtree(clf, features):

    # 重要度の抽出
    feature_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
    feature_importances = feature_importances.to_frame(name='重要度').sort_values(by='重要度', ascending=False)

    # TOP20可視化
    feature_importances[0:20].sort_values(by='重要度').plot.barh()
    plt.legend(loc='lower right')
    # plt.show()
    st.pyplot(plt)

    # # 重要度の抽出
    # feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
    # feature_importances = feature_importances.to_frame(name='重要度').sort_values(by='重要度', ascending=False)

    # # TOP20可視化
    # feature_importances[0:20].sort_values(by='重要度').plot.barh()
    # plt.legend(loc='lower right')
    # # plt.show()
    # st.pyplot(plt)



def ml_rtree(
    X: pd.DataFrame,
    y: pd.Series) -> list:

    # 学習
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)

    # 予測
    pred = clf.predict(X)

    # accuracyで精度評価
    score = accuracy_score(y, pred)

    return [clf, pred, score]


def ml_rtree_pred(
    X: pd.DataFrame,
    y: pd.Series,
    t_size: float) -> list:

    # train_test_split関数を利用してデータを分割する
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, train_size=t_size, random_state=0, stratify=y)

    # 学習
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_x, train_y)
    clf.fit(X, y)

    # 訓練データで予測 ＆ 精度評価
    train_pred = clf.predict(train_x)
    train_score = accuracy_score(train_y, train_pred)

    # 訓練データで予測 ＆ 精度評価
    valid_pred = clf.predict(valid_x)
    valid_score = accuracy_score(valid_y, valid_pred)

    return [clf, train_pred, train_score, valid_pred, valid_score]


def ml_rtree(
    X: pd.DataFrame,
    y: pd.Series) -> list:

    # 学習
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)

    # 予測
    pred = clf.predict(X)

    # accuracyで精度評価
    score = accuracy_score(y, pred)

    return [clf, pred, score]


def main():
    """ メインモジュール
    """

    # stのタイトル表示
    st.title("Simple AutoML Demo\n（Maschine Learning)")

    # ファイルのアップローダー
    uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv') 

    # サイドメニューの設定
    activities = ["データ確認", "要約統計量", "グラフ表示", "学習と検証", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'データ確認':
        # アップロードの有無を確認
        if uploaded_file is not None:

            # データフレームの読み込み
            df = pd.read_csv(uploaded_file)

            # ary_cnt = ["10", "50", "100", ]
            # cnt = st.sidebar.selectbox("Select Max mm", ary_cnt)
            cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

            # テーブルの表示
            st_display_table(df.head(int(cnt)))

        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '要約統計量':
        # アップロードの有無を確認
        if uploaded_file is not None:

            # データフレームの読み込み
            df = pd.read_csv(uploaded_file)

            # テーブルの表示
            st_display_table(df.describe())


    if choice == 'グラフ表示':
        # アップロードの有無を確認
        if uploaded_file is not None:

            # データフレームの読み込み
            df = pd.read_csv(uploaded_file)

            # ary_graph = ["ヒストグラム", "カウントプロット" ]
            # graph = st.sidebar.selectbox("グラフの種類", ary_graph)

            hue_col = df.columns[0]     # '退職'
            x_col = st.sidebar.selectbox("グラフのX軸", df.columns[0:])

            # ヒストグラムの表示
            st_display_histogram(df, x_col, 'null')


    if choice == '学習と検証':
        # アップロードの有無を確認
        if uploaded_file is not None:

            # データフレームの読み込み
            df = pd.read_csv(uploaded_file)

            # 説明変数と目的変数の設定
            train_X = df.drop("退職", axis=1) 
            train_Y = df["退職"]

            ary_algorithm = ["決定木", "ランダムフォレスト" ]
            algorithm = st.sidebar.selectbox("学習の手法", ary_algorithm)

            pred_flg = st.sidebar.selectbox('決定木の深さ', ['検証なし', '検証あり'])

            if algorithm == '決定木':
                depth = st.sidebar.number_input('決定木の深さ', min_value = 1, max_value = 5)
 
                # 決定木による予測
                # clf, pred, score = ml_dtree(train_X, train_Y, depth)
                # 決定木による予測
                clf, train_pred, train_score, valid_pred, valid_score = ml_dtree_pred(train_X, train_Y, depth, 2/3)


                st.subheader(f"訓練用データでの予測精度は")
                st.subheader(f"{train_score} でした。")

                # 特徴量の設定（決定木の可視化用）
                features = df.columns[1:]

                # 決定木の可視化
                st.caption('決定木の可視化')
                st_display_dtree(clf, features)

                if pred_flg == '検証あり':
                    st.subheader(f"検証用データでの予測精度は")
                    st.subheader(f"{valid_score} でした。")

            if algorithm == 'ランダムフォレスト':
                # ランダムフォレストによる予測
                # clf, pred, score = ml_rtree(train_X, train_Y)
                clf, train_pred, train_score, valid_pred, valid_score = ml_rtree_pred(train_X, train_Y, 2/3)

                st.subheader(f"訓練用データでの予測精度は")
                st.subheader(f"{train_score} でした。")

                # 特徴量の設定（重要度の可視化用）
                features = df.columns[1:]

                # 重要度の可視化
                st.caption('重要度の可視化')
                st_display_rtree(clf, features)

                if pred_flg == '検証あり':
                    st.subheader(f"検証用データでの予測精度は")
                    st.subheader(f"{valid_score} でした。")


    if choice == 'About':

        image = Image.open('logo_nail.png')
        st.image(image)

        # img_target = cv2.imread('logo_nail.png',  flags=cv2.IMREAD_COLOR)
        # st.image(img_target, width=600, channels='BGR',use_column_width=bool)

        #components.html("""""")
        st.markdown("Built by [Nail Team]")
        st.text("Version 0.1")
        st.markdown("For More Information check out   (https://nai-lab.com/)")
        

if __name__ == "__main__":
    main()

    # # stのタイトル表示
    # st.title("Iris データセットで Streamlit をお試し")

    # # ファイルのアップローダー（サイドバー）
    # uploaded_file = st.sidebar.file_uploader("ファイルアップロード", type='csv') 

    # if uploaded_file is not None:

    #     df = pd.read_csv(uploaded_file)

    #     st_display_df(df)
    #     st_display_table(df.head(10))
    #     st_display_berchart(df['年齢'])

    #     option = st.selectbox(
    #         'How would you like to be contacted?',
    #         ('Email', 'Home phone', 'Mobile phone'))

    #     st.write('You selected:', option)


    #     option2 = st.selectbox(
    #         'How would you like to be contacted?',
    #         df.columns)

    #     st.write('You selected:', option2)

    #     if st.button('列を選んて押してね'):
    #         st_display_berchart(df[option2])


