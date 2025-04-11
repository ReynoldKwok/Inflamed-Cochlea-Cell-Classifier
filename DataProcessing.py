import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from minisom import MiniSom
from collections import Counter


class Processing:

    def __init__(self):
        self.df_summary, self.class_df, self.plot_df, self.analyseDF = None, None, None, None
        self.raw_df, self.result_df, self.train_df = None, None, None
        self.class_list, self.label_list = [], []
        self.wave_no, self.start, self.end, self.iteration = 0, 0, 0, 0
        self.pca, self.som = None, None
        self.path = None
        self.som_fig, self.som_header, self.som_report, self.som_matrix = None, None, None, None
        self.X, self.y, self.X_train, self.X_test = None, None, None, None
        self.y_train, self.y_test, self.train_ids, self.test_ids = None, None, None, None

    def read_excel(self, path):
        self.path = path
        df = pd.read_excel(path, header=1)
        df = df.rename(columns={'Unnamed: 0': 'Dosage'})
        df1 = df.copy()
        # Add a column for specifying the dosage amount and rearrange the order of the dataset
        df1['Dosage amount'] = [int(x.split('ng')[0]) for x in df1['Dosage']]
        df1 = df1.sort_values(by='Dosage amount')
        row_no = len(df1)
        sample_groups = df1['Dosage'].value_counts(sort=False)
        summary_df = pd.DataFrame(sample_groups)
        # Get the data types of the columns in df4
        column_dtypes = df1.dtypes
        columns_no = (column_dtypes == 'float64').sum()
        columns_name = column_dtypes[column_dtypes == 'float64'].index
        self.wave_no = columns_name.values.astype(float)
        self.start = min(self.wave_no)
        self.end = max(self.wave_no)
        text = "Summary of the dataset:\n\n"
        text += f"File Name: {path.split('/')[-1]}\n\n"
        text += f"Number of samples: {row_no}\n"
        text += f"Number of variables: {columns_no} wavenumbers\n"
        text += f"Range of wavenumbers: {self.start} to {self.end}"
        class_list, class_amount = [], []
        for index, row in summary_df.iterrows():
            class_list.append(index)
            class_amount.append(row['count'])
        self.class_df = pd.DataFrame(columns=['Sample Group', 'Sample Amount'])
        self.class_df['Sample Group'] = class_list
        self.class_df['Sample Amount'] = class_amount
        self.class_df = self.class_df.set_index('Sample Group')
        self.df_summary = text
        self.class_list = class_list
        labels = [self.class_list.index(x) for x in df1['Dosage'].values]
        # Create a new column with the group labels
        df1['Dosage group'] = labels
        self.raw_df = df1.copy()
        self.raw_df = self.raw_df.reset_index(drop=True)
        df2 = df1.drop(['Dosage', 'Dosage amount'], axis=1).copy()
        # Calculate the mean, min, and max for each column in each subgroup
        subgroup_stats = df2.groupby('Dosage group').agg(['min', 'mean', 'max'])
        # Transform the subgroup_stats into a new dataframe for easier handling in visualisation
        df3 = pd.DataFrame(columns=df2.columns)
        df3.insert(0, 'Dosage', np.nan)
        df3.insert(1, 'Values', np.nan)
        df3 = df3.drop('Dosage group', axis=1)
        for dosage in range(len(self.class_list)):
            for value in range(3):
                content = [int(dosage), int(value)]
                for column in range(value, len(subgroup_stats.columns), 3):
                    content.append(subgroup_stats.iloc[dosage, column])
                df3.loc[len(df3)] = content
        df4 = df3.copy()
        df4.insert(0, 'Sample', '')
        for i, dosage in enumerate(df4['Dosage']):
            df4.loc[i, 'Sample'] = df1.loc[df1['Dosage group'] == dosage, 'Dosage'].values[0].split('Cell')[0]
        for name in df4['Sample'].unique():
            self.label_list.append(name)
        self.plot_df = df4.copy()
        return self.df_summary

    def plot(self, list1=None, list2=None):
        if list1 is None:
            list1 = [0, 1, 2, 3, 4, 5]
        if list2 is None:
            list2 = [1]
        df = self.plot_df.copy()
        # User inputs for the visualisation
        show_dosage = list1
        show_value = list2
        fill_area = 1

        # Default settings of the visualisation
        colours = ['blue', 'red', 'green', 'magenta', 'brown', 'gray', 'orange', 'pink', 'yellow']
        lineTypes = ['-', '--', ':', '-.', (5, (10, 3)), (0, (3, 1, 1, 1, 1, 1))]
        transparency = 0.3
        datalist = [[], [], []]
        old_dosage = -1
        plt.close('all')
        # Create a line plot for each statistic
        fig, ax = plt.subplots(figsize=(10, 4))
        linelist = []

        for row in range(len(df)):
            name = df.iloc[row, 0]
            dosage = df.iloc[row, 1]
            colour_index = int(dosage % len(colours))  # Cycle through the colors
            if dosage not in show_dosage:
                continue
            value = df.iloc[row, 2]
            if value not in show_value:
                continue
            data = df.iloc[row, 3:]
            if dosage != old_dosage:
                if value != 1:
                    line = ax.plot(self.wave_no, data, label=name, color=colours[colour_index], alpha=transparency,
                                   linestyle=lineTypes[int(dosage)])
                else:
                    line = ax.plot(self.wave_no, data, label=name, color=colours[colour_index],
                                   linestyle=lineTypes[int(dosage)])
            else:
                if value != 1:
                    line = ax.plot(self.wave_no, data, color=colours[colour_index], alpha=transparency,
                                   linestyle=lineTypes[int(dosage)])
                else:
                    line = ax.plot(self.wave_no, data, color=colours[colour_index], linestyle=lineTypes[int(dosage)])
            linelist.append(line)
            if fill_area == 1 and len(show_value) > 1:
                datalist[int(value)] = data.values.astype(float)
                if value == show_value[-1]:
                    ax.fill_between(self.wave_no, datalist[min(show_value)], datalist[max(show_value)],
                                    color=colours[colour_index], alpha=0.1)
            old_dosage = dosage

        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Intensity')
        ax.set_title('Visualisation of spectroscopy data')
        ax.legend()

        return fig

    def pca1(self):
        df = self.raw_df.dropna(axis=1).copy()
        scaler = StandardScaler()
        df1 = df.reset_index(drop=True)
        df1.columns = df1.columns.astype(str)
        X_scaled = scaler.fit_transform(df1.iloc[:, 1:-2])
        self.pca = PCA()
        self.pca.fit(X_scaled)
        # Create the scree plot
        plt.close('all')
        fig = plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), self.pca.explained_variance_ratio_)
        plt.xlabel('Number of principal components')
        plt.ylabel('Proportion of variance explained')
        plt.title('PCA: Scree Plot')
        plt.grid()

        explained_variance_ratio = self.pca.explained_variance_ratio_
        top_pc_indices = np.argsort(explained_variance_ratio)[::-1]
        text = "Top 5th Principal Components:\n"
        for i in top_pc_indices[0:5]:
            text += f"PC{i + 1}: {explained_variance_ratio[i] * 100:.2f}% of variance explained\n"
        text += '\n'
        return fig, text

    def pca2(self, list1=None):
        if list1 is None:
            list1 = [0, 1, 2, 3, 4]
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 4))
        # Plot the PCA loading spectra
        colours = ['blue', 'red', 'green', 'magenta', 'brown', 'gray', 'orange', 'pink', 'yellow']
        lineTypes = ['-', '--', ':', '-.', (5, (10, 3)), (0, (3, 1, 1, 1, 1, 1))]
        for i in list1:
            ax.plot(self.wave_no, self.pca.components_[i], label=f'PC{i + 1}', color=colours[i],
                    linestyle=lineTypes[i])
        plt.title('PCA: Loading Plot')
        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Loading')
        ax.legend()

        return fig, ax

    def data_select(self, ranges):
        processDF = self.raw_df.copy()
        if len(ranges) > 0:
            colIndex = []
            for i, subList in enumerate(ranges):
                for item in subList:
                    colList = processDF.columns[1:-2]
                    nearest = min(colList, key=lambda x: abs(float(x) - item))
                    index = processDF.columns.get_loc(nearest)
                    colIndex.append(index)

            for i, index in enumerate(colIndex):
                if i == 0:
                    processDF.iloc[:, colIndex[i]:-2] = np.nan
                elif i == len(colIndex) - 1:
                    processDF.iloc[:, 1:colIndex[i]] = np.nan
                else:
                    if i % 2 == 0:
                        processDF.iloc[:, colIndex[i]:colIndex[i - 1]] = np.nan
        self.analyseDF = processDF.copy()
        labels = [self.class_list.index(x) for x in self.analyseDF['Dosage'].values]
        self.analyseDF['Dosage group'] = labels

    def SOM(self, input_list=None, state=0):
        if input_list is None:
            input_list = [6, 6, 1.0, 0.5, 100, 80, 20]
        som_size = (input_list[0], input_list[1])  # Dimensions of the SOM grid
        sigma = input_list[2]  # Radius of the neighborhood function
        learning_rate = input_list[3]  # Learning rate

        # Create the SOM
        if state == input_list[4] // 5:
            self.som = MiniSom(som_size[0], som_size[1], self.X_train.shape[1], sigma=sigma,
                               learning_rate=learning_rate, random_seed=42)
            self.iteration = 0
        num_iterations = input_list[4]
        self.som.train_random(self.X_train, input_list[4] // 5, verbose=True)
        self.iteration += input_list[4] // 5
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'SOM Iteration - {self.iteration}')
        markers = ['o', 's', 'D', '^', '*', '+']
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

        w_x, w_y = zip(*[self.som.winner(d) for d in self.X_train])
        w_x = np.array(w_x)
        w_y = np.array(w_y)

        im1 = ax1.pcolor(self.som.distance_map().T, cmap='bone_r')
        plt.colorbar(im1)

        for c in np.unique(self.y_train):
            idx_target = self.y_train == c
            ax1.scatter(w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .6,
                        w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .6,
                        s=50, c=colors[c], label=self.label_list[c])

        ax1.legend(loc='upper left', bbox_to_anchor=(-0.45, 1))
        ax1.grid()

        im2 = ax2.pcolor(self.som.distance_map().T, cmap='bone_r')
        plt.colorbar(im2)
        w_list = []
        for cnt, xx in enumerate(self.X_train):
            w = self.som.winner(xx)  # getting the winner
            w_list.append([int(w[0]), int(w[1]), int(self.y_train[cnt])])

        votes = [[Counter() for _ in range(som_size[1])] for _ in range(som_size[0])]

        for x, y, label in w_list:
            votes[x][y][label] += 1

        label_list = []
        for y in range(som_size[1]):
            for x in range(som_size[0]):
                list1 = votes[x][y].most_common()
                max_count = 0
                max_count_label = []
                for i in list1:
                    if i[1] > max_count:
                        max_count = i[1]
                        max_count_label.append(i[0])
                    elif i[1] == max_count:
                        max_count_label.append(i[0])
                label_list.append([x, y, max_count_label])
        for grid in label_list:
            if len(grid[2]) > 1:
                for label in grid[2]:
                    ax2.plot(grid[0] + .5, grid[1] + .5, markers[label], markerfacecolor='None',
                             markeredgecolor=colors[label], markersize=8, markeredgewidth=1)
            elif len(grid[2]) > 0:
                ax2.plot(grid[0] + .5, grid[1] + .5, markers[grid[2][0]], markerfacecolor='None',
                         markeredgecolor=colors[grid[2][0]], markersize=8, markeredgewidth=1)
        unique_labels = np.unique(self.y_train)
        # Plot the legend
        for label in unique_labels:
            ax2.plot([], [], markers[label], markerfacecolor='None',
                     markeredgecolor=colors[label], label=self.label_list[label])
        ax2.legend(loc='upper right', bbox_to_anchor=(1.6, 1))
        ax2.grid()

        if self.iteration == num_iterations:
            winmap = self.som.labels_map(self.X_train, self.y_train)
            default_class = np.sum(list(winmap.values())).most_common()[0][0]
            y_pred = []
            for d in self.X_test:
                win_position = self.som.winner(d)
                if win_position in winmap:
                    y_pred.append(int(winmap[win_position].most_common()[0][0]))
                else:
                    y_pred.append(int(default_class))

            result = classification_report(self.y_test, y_pred)
            report, header = self.cr(result)
            accuracy = confusion_matrix(self.y_test, y_pred)
            matrix = self.cm(accuracy)
            pred = []
            for x in y_pred:
                pred.append(self.label_list[x])
            y_pred_df = pd.DataFrame({'y_pred': pred}, index=self.test_ids)
            result_df = self.result_df.merge(y_pred_df, left_index=True, right_index=True, how='left')
            self.result_df['SOM Predicted'] = result_df['y_pred']
            self.save_som(fig, header, report, matrix)
        else:
            header = None
            report = None
            matrix = None

        return fig, header, report, matrix

    def knn(self, para=None):
        if para is None:
            para = [5]
        knn = KNeighborsClassifier(n_neighbors=para[0])
        knn.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred = knn.predict(self.X_test)
        result = classification_report(self.y_test, y_pred)
        report, header = self.cr(result)
        accuracy = confusion_matrix(self.y_test, y_pred)
        matrix = self.cm(accuracy)
        pred = []
        for x in y_pred:
            pred.append(self.label_list[x])
        y_pred_df = pd.DataFrame({'y_pred': pred}, index=self.test_ids)
        y_pred_df.to_csv('label.csv')
        result_df = self.result_df.merge(y_pred_df, left_index=True, right_index=True, how='left')
        self.result_df['KNN Predicted'] = result_df['y_pred']
        return header, report, matrix

    def dt(self, para=None):
        if para is None:
            para = [5]
        dt = DecisionTreeClassifier(criterion='gini', max_depth=para[0], random_state=42)
        dt.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred = dt.predict(self.X_test)
        result = classification_report(self.y_test, y_pred)
        report, header = self.cr(result)
        accuracy = confusion_matrix(self.y_test, y_pred)
        matrix = self.cm(accuracy)
        pred = []
        for x in y_pred:
            pred.append(self.label_list[x])
        y_pred_df = pd.DataFrame({'y_pred': pred}, index=self.test_ids)
        result_df = self.result_df.merge(y_pred_df, left_index=True, right_index=True, how='left')
        self.result_df['Decision Tree Predicted'] = result_df['y_pred']
        return header, report, matrix

    def rf(self, para=None):
        if para is None:
            para = [100, 5]
        rf = RandomForestClassifier(n_estimators=para[0], criterion='gini', max_depth=para[1], random_state=42)
        rf.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred = rf.predict(self.X_test)
        result = classification_report(self.y_test, y_pred)
        report, header = self.cr(result)
        accuracy = confusion_matrix(self.y_test, y_pred)
        matrix = self.cm(accuracy)
        pred = []
        for x in y_pred:
            pred.append(self.label_list[x])
        y_pred_df = pd.DataFrame({'y_pred': pred}, index=self.test_ids)
        result_df = self.result_df.merge(y_pred_df, left_index=True, right_index=True, how='left')
        self.result_df['Random Forest Predicted'] = result_df['y_pred']
        return header, report, matrix

    def xg(self, para=None):
        if para is None:
            para = [100, 5, 0.1]
        xg = xgb.XGBClassifier(
            objective='multi:softmax',
            max_depth=para[1],
            n_estimators=para[0],
            learning_rate=para[2],
            random_state=42
        )
        xg.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred = xg.predict(self.X_test)
        result = classification_report(self.y_test, y_pred)
        report, header = self.cr(result)
        accuracy = confusion_matrix(self.y_test, y_pred)
        matrix = self.cm(accuracy)
        pred = []
        for x in y_pred:
            pred.append(self.label_list[x])
        y_pred_df = pd.DataFrame({'y_pred': pred}, index=self.test_ids)
        result_df = self.result_df.merge(y_pred_df, left_index=True, right_index=True, how='left')
        self.result_df['XG Boost Predicted'] = result_df['y_pred']
        return header, report, matrix

    def svm(self, para=None):
        if para is None:
            para = [1]
        svm = SVC(kernel='rbf', C=para[0], gamma='scale', random_state=42)
        svm.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        y_pred = svm.predict(self.X_test)
        result = classification_report(self.y_test, y_pred)
        report, header = self.cr(result)
        accuracy = confusion_matrix(self.y_test, y_pred)
        matrix = self.cm(accuracy)
        pred = []
        for x in y_pred:
            pred.append(self.label_list[x])
        y_pred_df = pd.DataFrame({'y_pred': pred}, index=self.test_ids)
        result_df = self.result_df.merge(y_pred_df, left_index=True, right_index=True, how='left')
        self.result_df['SVM Predicted'] = result_df['y_pred']
        return header, report, matrix

    def cm(self, matrix=None):
        cm = pd.DataFrame(matrix, columns=self.label_list)
        cm['Dosage'] = self.label_list
        cm = cm.set_index('Dosage')
        return cm

    def cr(self, accuracy=None):
        text_list = accuracy.split()
        report = pd.DataFrame(columns=["Dosage", "Precision", "Recall", "F1-Score", "Support"])
        report['Dosage'] = self.label_list
        start = text_list.index('support') + 1
        end = text_list.index('accuracy')
        score_list = text_list[start:end]
        precision_list, recall_list, f1_list, support_list = [], [], [], []
        for i in range(0, len(score_list), 5):
            precision_list.append(score_list[i + 1])
            recall_list.append(score_list[i + 2])
            f1_list.append(score_list[i + 3])
            support_list.append(score_list[i + 4])
        report['Precision'] = precision_list
        report['Recall'] = precision_list
        report['F1-Score'] = f1_list
        report['Support'] = support_list
        report = report.set_index('Dosage')
        header = [f'No. of testing data: {text_list[end + 2]}', f'Accuracy: {text_list[end + 1]}']
        return report, header

    def pred_result(self, test_list=None, pred_list=None):
        result = pd.DataFrame(test_list, columns=['Actual\nLabel'])
        result['Predicted\n   Label'] = pred_list
        result = result.set_index('Actual\nLabel')
        result = result.sort_index()
        return result

    def get_range(self):
        return self.start, self.end

    def split_data(self, gui=None, size=0.2, state=0, train_df=None):
        self.result_df = self.raw_df.copy()
        self.result_df.drop(['Dosage amount', 'Dosage group'], axis=1)
        self.result_df.insert(1, 'SVM Predicted', ['N/A'] * self.result_df.shape[0])
        self.result_df.insert(1, 'XG Boost Predicted', ['N/A'] * self.result_df.shape[0])
        self.result_df.insert(1, 'Random Forest Predicted', ['N/A'] * self.result_df.shape[0])
        self.result_df.insert(1, 'Decision Tree Predicted', ['N/A'] * self.result_df.shape[0])
        self.result_df.insert(1, 'KNN Predicted', ['N/A'] * self.result_df.shape[0])
        self.result_df.insert(1, 'SOM Predicted', ['N/A'] * self.result_df.shape[0])
        gui.ml_state = 1
        if state == 0:
            df = self.analyseDF.iloc[:, 1:-2].dropna(axis=1).copy()
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(df), columns=df.columns.astype(str))
            self.X = X.values
            # Add a new column with the sample IDs
            X['sample_id'] = self.analyseDF.index
            self.y = self.analyseDF['Dosage group'].values
            self.train_df = self.analyseDF.iloc[:, 0:-2].dropna(axis=1).copy()
            gui.split = size
            self.X_train, self.X_test, self.y_train, self.y_test, self.train_ids, self.test_ids = train_test_split(
                self.X, self.y, X['sample_id'], test_size=size)
            self.train_df = self.train_df.drop(self.train_df.index[self.test_ids])
        else:
            df = self.analyseDF.iloc[:, 0:-2].dropna(axis=1).copy()
            df.columns = df.columns.astype(str)
            df1 = train_df.copy()
            df1.columns = df1.columns.astype(str)
            df_len = len(df)
            if len(df.columns) != len(df1.columns):
                return 1
            df_concat = pd.concat([df, df1], axis=0, ignore_index=True)
            scaler = StandardScaler()
            X = df_concat.iloc[:, 1:].copy()
            y = df_concat.iloc[:, 0:1].copy()
            norm_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns.astype(str))
            labels = [self.class_list.index(x) for x in y.values]
            self.X_train = np.array(norm_X.values[df_len:])
            self.X_test = np.array(norm_X.values[:df_len])
            self.y_train = np.array(labels[df_len:])
            self.y_test = np.array(labels[:df_len])
            self.test_ids = df.index

    def load_ml(self, gui=None, features=None, df=None):
        self.data_select(features)
        errorno = self.split_data(gui=gui, state=1, train_df=df)
        return errorno

    def save_som(self, fig=None, header=None, report=None, matrix=None):
        self.som_fig = fig
        self.som_header = header
        self.som_report = report
        self.som_matrix = matrix

    def get_info(self):
        num_rows, num_cols = self.X_train.shape
        return num_rows, num_cols

    def get_summary(self):
        return self.df_summary, self.class_df

    def get_list(self):
        return self.class_list, self.label_list

    def get_ml(self):
        return self.train_df

    def get_features(self):
        return [[float(self.start), float(self.end)]]


def sortList(sublist=None):
    sublist = sorted(sublist, key=lambda x: x[0])
    for i, list1 in enumerate(sublist):
        if i != len(sublist) - 1:
            list2 = sublist[i + 1]
            state, start, end = checkOverlap(list1, list2)
            if state == 0:
                sublist.remove(list1)
            elif state == 1:
                sublist.remove(list1)
                sublist.remove(list2)
                sublist.insert(i, [start, end])
        else:
            list2 = sublist[i - 1]
            state, start, end = checkOverlap(list2, list1)
            if state == 0:
                sublist.remove(list2)
            elif state == 1:
                sublist.remove(list1)
                sublist.remove(list2)
                sublist.insert(i, [start, end])
    return sublist


def checkOverlap(list1, list2):
    if (list2[0] <= list1[0] <= list2[1]) or (list2[0] <= list1[1] <= list2[1]):
        if list1[0] >= list2[0]:
            if list1[1] <= list2[1]:
                return 0, list2[0], list2[1]
            elif list1[1] > list2[1]:
                return 1, list2[0], list1[1]
        elif list1[0] <= list2[0]:
            if list1[1] <= list2[1]:
                return 1, list1[0], list2[1]
            elif list1[1] > list2[1]:
                return 2, list1[0], list1[1]
    else:
        if list1[0] >= list2[1] or list1[1] <= list2[0]:
            return 2, list1[0], list1[1]
        elif list1[0] <= list2[0] and list1[1] >= list2[1]:
            return 1, list1[0], list1[1]
