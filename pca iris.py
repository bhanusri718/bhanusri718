from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
x=iris.data
y=iris.target
pca =PCA(n_components=2)
X_pca = pca.fit_transform(x)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:,1],
                hue=y, palette='viridis',s=50)
plt.title('PCA: Iris dataset')
plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.legend()
plt.show()

