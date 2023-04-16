import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder


from typing import List, Dict

"""
Constants
"""

COMPRESSION_GZIP = "gzip"
DIAGNOSIS_DATA_PATH = "data/MIMIC3_DIAGNOSES_ICD_subset.csv.gz"
DESC_DATA_PATH = "data/D_ICD_DIAGNOSES.csv.gz"

COLUMN_SUBJECT_ID = "SUBJECT_ID"
COLUMN_ICD9_CODE = "ICD9_CODE"
COLUMN_SHORT_TITLE = "SHORT_TITLE"

K = 5  # Number of topics
TOTAL_EPOCHS = 100
TOP_ICD_COUNT = 10
TOP_PATIENT_COUNT = 100

ALPHA = 1  # Document topic Dirichlet prior
BETA = 0.001  # ICD-9 topic Dirichlet prior


"""
ICD Code Data Processor
"""


class ICDCodeDataProcessor:

    """ Public methods """

    def create_patient_docs(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """
        @param self:
        @param df: encoded diagnosis dataframe
        @return: patient docs
        """

        result: Dict[int, List[int]] = None

        if df is not None:

            result = dict()
            for _, row in df.iterrows():

                patient_id = row[COLUMN_SUBJECT_ID]
                doc = result.get(patient_id)
                if doc is None:
                    doc: List[int] = list()

                icd_code = row[COLUMN_ICD9_CODE]
                doc.append(icd_code)
                result[patient_id] = doc

        return result


"""
Plot
"""


class ICDPlot:

    """ Public methods """

    def plot_top_icds(self, descriptions: pd.DataFrame,
                      phi: np.array, topic_labels: List[str],
                      icd_encoder: LabelEncoder) -> None:

        top_icds = np.zeros((K * TOP_ICD_COUNT, K))
        top_icds_labels = []
        for k in range(K):

            # Top ICD codes
            top_idx = np.argsort(phi[:, k])[::-1][:TOP_ICD_COUNT]
            top_value = np.take(phi[:, k], top_idx)
            top_icds[k * TOP_ICD_COUNT: (k + 1) * TOP_ICD_COUNT, k] = top_value

            # Top labels
            icd_codes = icd_encoder.inverse_transform(top_idx)
            if icd_codes is not None and icd_codes.shape[0] > 0:

                for icd_code in icd_codes:

                    icd_code_value = str(icd_code)
                    description_row = descriptions.loc[descriptions[COLUMN_ICD9_CODE].str.startswith(
                        icd_code_value)]

                    if description_row.empty:
                        continue

                    description_row = description_row.iloc[0]
                    short_title = description_row[COLUMN_SHORT_TITLE]
                    label = icd_code_value + "-" + short_title

                    top_icds_labels.append(label)

        assert len(top_icds_labels) == K * TOP_ICD_COUNT

        plt.clf()
        plt.figure(figsize=(3, 10))
        plt.title("Latent topics inferred from the collapsed Gibbs sampling")
        plt.tight_layout()
        axs = sns.heatmap(top_icds, cmap="Reds",
                          yticklabels=top_icds_labels,
                          xticklabels=topic_labels)
        cbar = axs.collections[0].colorbar
        cbar.ax.set_title("topic prob", fontsize=9)
        plt.savefig("figure_1_top_icds.png", bbox_inches='tight')
        plt.close()

    def plot_correlation(self, D: int, theta: np.array,
                         patient_docs: Dict[int, List[int]],
                         topic_labels: List[str],
                         icd_xticklabels: List[str],
                         icd_encoder: LabelEncoder) -> np.array:

        target_icd_codes = np.zeros((D, 3))
        for d, doc in enumerate(patient_docs.values()):

            icd_codes = icd_encoder.inverse_transform(doc)
            if icd_codes is not None and icd_codes.shape[0] > 0:

                for icd_code in icd_codes:

                    icd_code_value = str(icd_code)
                    if icd_code_value.startswith("331"):
                        target_icd_codes[d, 0] = 1

                    elif icd_code_value.startswith("332"):
                        target_icd_codes[d, 1] = 1

                    elif icd_code_value.startswith("340"):
                        target_icd_codes[d, 2] = 1

        total_correlation = np.corrcoef(theta.T, target_icd_codes.T)  # 8 x 8

        # First 5 rows are correlation from 5 topics
        # Last 3 columns are correlation from 3 ICD codes
        correlation = total_correlation[:5, -3:]

        plt.clf()
        plt.figure(figsize=(6, 5))
        plt.title("Topics by target ICD correlation")
        plt.tight_layout()
        axs = sns.heatmap(correlation, cmap="coolwarm",
                          yticklabels=topic_labels,
                          xticklabels=icd_xticklabels)
        cbar = axs.collections[0].colorbar
        cbar.ax.set_title("ICD-topic cor", fontsize=9)
        plt.yticks(rotation=0)
        plt.savefig("figure_2_correlation.png", bbox_inches='tight')
        plt.close()

        return target_icd_codes

    def plot_top_patients(self, theta: np.array, target_icd_codes: np.array,
                          topic_labels: List[str],
                          icd_xticklabels: List[str]) -> None:

        top_patients_idx = np.zeros((K * TOP_PATIENT_COUNT)).astype(int)
        for k in range(K):
            # Top patients idx
            top_idx = np.argsort(theta[:, k])[::-1][:TOP_PATIENT_COUNT]
            top_patients_idx[k *
                             TOP_PATIENT_COUNT: (k + 1) * TOP_PATIENT_COUNT] = top_idx

        top_patients = theta[top_patients_idx]
        top_target_icd_codes = target_icd_codes[top_patients_idx]

        cmap_1 = LinearSegmentedColormap.from_list(
            "custom", ["#FFFFFF", "#000000"])
        cmap_2 = LinearSegmentedColormap.from_list(
            "custom", ["#FFFFFF", "#FF0000"])
        cmap_3 = LinearSegmentedColormap.from_list(
            "custom", ["#FFFFFF", "#0000FF"])
        cmaps = [cmap_1, cmap_2, cmap_3]
        plt.clf()
        fig, axs = plt.subplots(nrows=1, ncols=top_target_icd_codes.shape[1] + 1,
                                width_ratios=[1, 1, 1, 12],
                                gridspec_kw={'wspace': 0},
                                figsize=(6, 10))
        fig.suptitle("Top 100 patients per topic")
        annotation_axs = axs[:3]
        for i, (a, c, l) in enumerate(zip(annotation_axs, cmaps, icd_xticklabels)):
            sns.heatmap(top_target_icd_codes[:, i][:, None],
                        ax=a, cmap=c,
                        xticklabels=[l],
                        yticklabels=False,
                        cbar=False)
            a.tick_params(labelrotation=90)

        top_patients_axs = axs[-1]
        sns.heatmap(top_patients, cmap="Reds",
                    ax=top_patients_axs,
                    yticklabels=False,
                    xticklabels=topic_labels)
        top_patients_axs.tick_params(labelrotation=90)
        cbar = top_patients_axs.collections[0].colorbar
        cbar.ax.set_title("topic prob", fontsize=9)
        fig.savefig("figure_3_top_patients.png", bbox_inches='tight')
        plt.close()


"""
LDA
"""


class LDAModel:

    """ Initialize """

    def __init__(self, K: int, D: int, M: int,
                 alpha: float, beta: float,
                 patient_docs: Dict[int, List[int]]) -> None:
        """
        @param self:
        @param K: Number of topics
        @param D: Number of unique patients (documents)
        @param M: Number of unique ICD codes (vocabulary)
        @param alpha: Hyper-parameter alpha to generate topic distribution
        @param beta: Hyper-parameter beta to generate word x topic matrix
        @param patient_docs: patient docs
        """

        self._K = K
        self._D = D
        self._M = M
        self._alpha = alpha
        self._beta = beta
        self._patient_docs = patient_docs

        self._initialize()

    """ Getters """

    @property
    def theta(self) -> np.array:
        if self._theta is None:
            raise Exception("Please run fit() first!")
        return self._theta

    @property
    def phi(self) -> np.array:
        if self._phi is None:
            raise Exception("Please run fit() first!")
        return self._phi

    """ Public methods """

    def fit(self, epochs: int) -> None:
        """
        @param self:
        @param epochs: Training epochs
        """

        for _ in range(epochs):

            for d, doc in enumerate(self._patient_docs.values()):
                for m in doc:

                    # Sample new topic
                    updated_topic = self._sample_topic(d, m)

                    # Update count
                    self._z[d, m] = updated_topic
                    self._doc_topic[d, updated_topic] += 1
                    self._topic_terms[updated_topic, m] += 1
                    self._terms_count_in_topic[updated_topic] += 1

        # Update theta and phi
        self._update_parameter()

    """ Private methods """

    def _initialize(self) -> None:

        # Topics
        self._z = np.zeros((self._D, self._M)).astype(
            int)  # Document x Vocabulary

        # The number of words assigned to topic k in document d
        self._doc_topic = np.zeros((self._D, self._K))  # Document x Topic

        # The number of words in document d
        self._terms_count_in_doc = np.zeros((self._D))  # Document x 1

        # The number of word m assigned to topic k in all documents
        self._topic_terms = np.zeros((self._K, self._M))  # Topic x Vocabulary

        # The number of words assigned to topic k in all documents
        self._terms_count_in_topic = np.zeros((self._K))  # Topics x 1

        # Topic distribution, Document x Topic
        self._theta = np.zeros((self._D, self._K))

        # Word x Topic
        self._phi = np.zeros((self._M, self._K))

        # Initialize
        for d, doc in enumerate(self._patient_docs.values()):

            for m in doc:

                rand_topic = np.random.randint(self._K)

                self._z[d, m] = rand_topic
                self._doc_topic[d, rand_topic] += 1
                self._topic_terms[rand_topic, m] += 1
                self._terms_count_in_topic[rand_topic] += 1

            # Total number of words in document d
            self._terms_count_in_doc[d] = len(doc)

    def _sample_topic(self, d: int, m: int) -> int:

        # No need to consider the current word
        old_topic = self._z[d, m]
        self._doc_topic[d, old_topic] -= 1
        self._topic_terms[old_topic, m] -= 1
        self._terms_count_in_topic[old_topic] -= 1

        # Compute p(z=k)
        probs = np.zeros((self._K))
        for k in range(self._K):

            first_term = self._alpha + self._doc_topic[d, k]
            numerator = self._beta + self._topic_terms[k, m]
            denominator = self._M * self._beta + self._terms_count_in_topic[k]
            second_terms = numerator / denominator
            probs[k] = first_term * second_terms

        # Normalize p(z=k)
        probs = probs / np.sum(probs)

        # Sample new topic
        sample = np.random.multinomial(1, probs)
        return np.nonzero(sample)[0][0]

    def _update_parameter(self) -> None:

        # Update topic distribution (theta)
        for d in range(len(self._patient_docs)):
            for k in range(self._K):

                numerator = self._alpha + self._doc_topic[d, k]
                denominator = self._K * self._alpha + \
                    self._terms_count_in_doc[d]

                self._theta[d, k] = numerator / denominator

        # Update word x topic matrix (phi)
        for m in range(self._M):
            for k in range(self._K):

                numerator = self._beta + self._topic_terms[k, m]
                denominator = self._M * self._beta + \
                    self._terms_count_in_topic[k]

                self._phi[m, k] = numerator / denominator


"""
Main
"""


def main():

    # Load data
    diagnosis = pd.read_csv(DIAGNOSIS_DATA_PATH,
                            compression=COMPRESSION_GZIP)
    print("Diagnosis shape: {} \n{}".format(diagnosis.shape, diagnosis.head()))

    descriptions = pd.read_csv(DESC_DATA_PATH,
                               compression=COMPRESSION_GZIP)
    print("Description shape: {} \n{}".format(
        descriptions.shape, descriptions.head()))

    D = len(diagnosis[COLUMN_SUBJECT_ID].unique())
    M = len(diagnosis[COLUMN_ICD9_CODE].unique())

    # Preprocess data
    patient_encoder = LabelEncoder()
    icd_encoder = LabelEncoder()

    encoded_patients = patient_encoder.fit_transform(
        diagnosis[COLUMN_SUBJECT_ID])
    encoded_icds = icd_encoder.fit_transform(diagnosis[COLUMN_ICD9_CODE])

    encoded_diagnosis = pd.DataFrame(
        columns=[COLUMN_SUBJECT_ID, COLUMN_ICD9_CODE])
    encoded_diagnosis[COLUMN_SUBJECT_ID] = encoded_patients
    encoded_diagnosis[COLUMN_ICD9_CODE] = encoded_icds

    data_processor = ICDCodeDataProcessor()
    patient_docs = data_processor.create_patient_docs(encoded_diagnosis)

    # Q1: Fit LDA model
    lda = LDAModel(K=K, D=D, M=M,
                   alpha=ALPHA, beta=BETA,
                   patient_docs=patient_docs)
    lda.fit(epochs=TOTAL_EPOCHS)

    phi = lda.phi
    theta = lda.theta

    # Q2: Visualise the top ICD codes under each topic
    icd_plot = ICDPlot()
    topic_labels = ["topic {}".format(k + 1) for k in range(K)]
    icd_plot.plot_top_icds(descriptions, phi, topic_labels, icd_encoder)

    # Q3: Correlating topics with the target ICD codes
    icd_xticklabels = ["icd331", "icd332", "icd340"]
    target_icd_codes = icd_plot.plot_correlation(D, theta, patient_docs,
                                                 topic_labels, icd_xticklabels, icd_encoder)

    # Q4: Visualizing patient topic mixtures
    icd_plot.plot_top_patients(theta, target_icd_codes,
                               topic_labels, icd_xticklabels)


if __name__ == "__main__":
    main()
