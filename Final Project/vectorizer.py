import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def debug(tuple_stuff):


    print('debugging')
    return


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    # if grp_ids:
    #     D = Xtr[grp_ids].toarray()
    # else:
    #     D = Xtr.toarray()
    if grp_ids == None:
        grp_ids = list(range(Xtr.shape[0]))

    tfidf_means = np.zeros([Xtr.shape[1]])
    for i in range(grp_ids[0].__len__()):
        val = grp_ids[0][i]
        D = Xtr[val, :].toarray()
        D[D < min_tfidf] = 0
        tfidf_means[i] = np.mean(D)
    print(tfidf_means.shape)
    # for i in range(grp_ids.__len__()):
    #     val = grp_ids[0][i]
    #     print('i: ' + str(i))
    #     print(Xtr[val].shape)
    #     print(tfidf_means.shape)
    #     D = Xtr[grp_ids[0][i]].toarray()
    #     D[D < min_tfidf] = 0
    #     print(np.mean(D, axis=1))
    #     tfidf_means[i] = np.mean(D, axis=0)

    # D[D < min_tfidf] = 0
    # tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=10):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


# In error analysis I make a train/test set from my cross validation set.
# Then I create a baseline error
# Looking for


# To create the horizontal and vertical differences I
# Are there other terms I can search for that can give me the "horizontal vs vertical difference".
# Normalize ranges to compare.
# For every feature compute horizontal and absolute

# When can we get the Midterm 2 result?

