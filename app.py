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

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 精度評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

# ロゴの表示用
from PIL import Image

# ディープコピー
import copy

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
            sns.countplot(data=df, x=x_col, ax=ax)
    else:
        sns.countplot(data=df, x=x_col, hue=hue_col, ax=ax)

    st.pyplot(fig)

    # seabornでグラフを複数のグラフを描画する - Qiita
    # https://qiita.com/tomokitamaki/items/b954e26be739bee5621e



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



def ml_drtree_pred(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm,
    depth: int,
    t_size: float) -> list:

    # train_test_split関数を利用してデータを分割する
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, train_size=t_size, random_state=0, stratify=y)

    # データを水増し（オーバーサンプリング）する
    oversample = SMOTE(sampling_strategy=0.5, random_state=0)
    train_x, train_y = oversample.fit_resample(train_x, train_y)


    if algorithm == 'dtree':
        # 分類器の設定
        clf = DecisionTreeClassifier(max_depth=depth)

    elif algorithm == 'rtree':
        # 分類器の設定
        clf = RandomForestClassifier(random_state=0)

    # 学習
    clf.fit(train_x, train_y)

    # 戻り値の初期化
    train_scores = []
    valid_scores = []

    # 訓練データで予測 ＆ 精度評価
    train_pred = clf.predict(train_x)
    
    if np.count_nonzero(train_pred == 'Yes') == 0:
        # 予測が全て'No'だった場合...
        train_scores = [0, 0, 0]
        valid_scores = [0, 0, 0]
        train_pred = np.nan
        valid_pred = np.nan
    else:
        # 目的変数を0,1に変換
        y_true = pd.get_dummies(train_y, drop_first=True)
        y_true = y_true['Yes'] # 型をSeriesに変換
        y_pred = pd.get_dummies(train_pred, drop_first=True)
        y_pred = y_pred['Yes'] # 型をSeriesに変換

        train_scores.append(round(accuracy_score(y_true, y_pred),3))
        train_scores.append(round(recall_score(y_true, y_pred),3))
        train_scores.append(round(precision_score(y_true, y_pred),3))

        # # 訓練データで予測 ＆ 精度評価
        valid_pred = clf.predict(valid_x)

        # 目的変数を0,1に変換
        y_true = pd.get_dummies(valid_y, drop_first=False)
        y_true = y_true['Yes'] # 型をSeriesに変換
        y_pred = pd.get_dummies(valid_pred, drop_first=False)
        y_pred = y_pred['Yes'] # 型をSeriesに変換

        valid_scores.append(round(accuracy_score(y_true, y_pred),3))
        valid_scores.append(round(recall_score(y_true, y_pred),3))
        valid_scores.append(round(precision_score(y_true, y_pred),3))

    return [clf, train_pred, train_scores, valid_pred, valid_scores]


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

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

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

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # テーブルの表示
                st_display_table(df.describe())


    if choice == 'グラフ表示':
        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # ary_graph = ["ヒストグラム", "カウントプロット" ]
                # graph = st.sidebar.selectbox("グラフの種類", ary_graph)

                hue_col = df.columns[0]     # '退職'
                x_col = st.sidebar.selectbox("グラフのX軸", df.columns[0:])

                # ヒストグラムの表示
                st_display_histogram(df, x_col, 'null')


    if choice == '学習と検証':
        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # 説明変数と目的変数の設定
                train_X = df.drop("退職", axis=1) 
                train_Y = df["退職"]

                ary_algorithm = ["決定木", "ランダムフォレスト" ]
                algorithm = st.sidebar.selectbox("学習の手法", ary_algorithm)

                # pred_flg = st.sidebar.selectbox('決定木の深さ', ['検証なし', '検証あり'])

                if algorithm == '決定木':
                    depth = st.sidebar.number_input('決定木の深さ (サーバの負荷軽減の為 Max=3)', min_value = 1, max_value = 3)
    
                    # 決定木による予測
                    clf, train_pred, train_scores, valid_pred, valid_scores = ml_drtree_pred(train_X, train_Y, 'dtree', depth, 2/3)


                    # 特徴量の設定（決定木の可視化用）
                    features = df.columns[1:]

                    # 決定木の可視化
                    st.caption('決定木の可視化')
                    st_display_dtree(clf, features)

                if algorithm == 'ランダムフォレスト':
                    # ランダムフォレストによる予測
                    clf, train_pred, train_scores, valid_pred, valid_scores = ml_drtree_pred(train_X, train_Y, 'rtree', 0, 2/3)

                    # 特徴量の設定（重要度の可視化用）
                    features = df.columns[1:]

                    # 重要度の可視化
                    st.caption('重要度の可視化')
                    st_display_rtree(clf, features)


                # 決定木＆ランダムフォレストの予測精度を表示
                st.subheader(f"訓練用データでの予測精度")
                st.caption('AIの予測が「全員、退職しない」に偏った場合は（意味がないので）全ての精度は0で表示されます')

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.text('正解率')
                    st.subheader(f"{train_scores[0]}")

                with col2:
                    st.text('再現率')
                    st.subheader(f"{train_scores[1]}")

                with col3:
                    st.text('適合率')
                    st.subheader(f"{train_scores[2]}")

                st.subheader(f"検証用データでの予測精度")

                col4, col5, col6 = st.columns(3)

                with col4:
                    st.text('正解率')
                    st.subheader(f"{valid_scores[0]}")

                with col5:
                    st.text('再現率')
                    st.subheader(f"{valid_scores[1]}")

                with col6:
                    st.text('適合率')
                    st.subheader(f"{valid_scores[2]}")


    if choice == 'About':

        image = Image.open('logo_nail.png')
        st.image(image)

        #components.html("""""")
        st.markdown("Built by [Nail Team]")
        st.text("Version 0.3")
        st.markdown("For More Information check out   (https://nai-lab.com/)")
        

if __name__ == "__main__":
    main()

