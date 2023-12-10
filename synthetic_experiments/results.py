#Output results as a heatmap

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

'''
data = np.array([
    [0.718, 0.452, 0.689, 0.688],
    [0.752, 0.343, 0.804, 1.522],
    [0.008, 0.72,  0.484, 1.062],
    [0.819, 0.671, 0.718, 1.382]
])
'''
'''
data = np.array([
    [0.053469104368021325, 0.05272878465316866, 3.170745462498715, 1.125691447009976, 0.05065927329996334],
    [2.9041168258136647, 2.9041168258136674, 2.088622350143262, 0.3459694690029953, 0.07138365757040377],
    [6.568170341355086, 6.568170341355086, 3.407499146031242, 0.058351105928105884, 0.13490616266024702],
    [0.3464091610962253, 0.3470745149020246, 2.844214491358272, 0.1396520317386427, 0.19982856020593515],
    [1.797562498594683, 2.8650076895387526, 1.2895540828653935, 1.2688054073626773, 0.05254400616770005],
    [0.1926079958987842, 0.25361258215064364, 1.788781339663366, 0.6023460216622049, 0.3168338212163536],
    [0.09570070197544145, 0.09565029174084083, 1.2436818275806054, 0.1998824535359751, 0.27263065476519033],
    [1.4676691264702488, 1.886335637072707, 0.02031542052264035, 2.581635451524263, 0.2778433459662767],
    [4.731406825135263, 2.7877675239272075, 4.702453013483652, 3.627080115724877, 4.522233036199511],
    [4.485616105900042, 4.713978903503316, 4.5142803069904005, 11.603401443903032, 11.136399702769415],
    [3.5238826720577276, 3.1725512930351956, 1.4618735562214153, 0.7108629769818866, 0.7844653054311939],
])
'''
'''
data = np.array([
    [5.410387919048975, 1.9888382297802611, 1.0085907396903164],
    [14.65042313900106, 1.6242805981539126, 2.3962809828652953],
    [3.1874760164819342, 2.0608260538233636, 0.776788528229501], 
    [np.nan, 2.1721308464144067, 2.262866589889685], 
    [3.992140762190113, 2.947600806343599, 1.5067386637438194], 
    [5.470204806078328, 2.3764742830051633, 1.2476799856724419],
    [5.426157234141189, 2.7863155755092466, 0.47377331452948535],
    [9.76549473142389, 13.135203841406177, 4.811414303958371],
    [6.874793495833543, 6.741931433350623, 9.53429139110998],
    [np.nan, 4.716221906356266, 2.705332173796982]
])
'''

data = np.array([
    [0.053469104368021325, 0.05272878465316866, 3.170745462498715, 1.125691447009976, 0.05065927329996334, 5.410387919048975, 1.9888382297802611, 1.0085907396903164],
    [2.9041168258136647, 2.9041168258136674, 2.088622350143262, 0.3459694690029953, 0.07138365757040377, np.nan, np.nan, np.nan],
    [6.568170341355086, 6.568170341355086, 3.407499146031242, 0.058351105928105884, 0.13490616266024702, 14.65042313900106, 1.6242805981539126, 2.3962809828652953],
    [0.3464091610962253, 0.3470745149020246, 2.844214491358272, 0.1396520317386427, 0.19982856020593515, 3.1874760164819342, 2.0608260538233636, 0.776788528229501],
    [1.797562498594683, 2.8650076895387526, 1.2895540828653935, 1.2688054073626773, 0.05254400616770005, np.nan, 2.1721308464144067, 2.262866589889685],
    [0.1926079958987842, 0.25361258215064364, 1.788781339663366, 0.6023460216622049, 0.3168338212163536, 3.992140762190113, 2.947600806343599, 1.5067386637438194],
    [0.09570070197544145, 0.09565029174084083, 1.2436818275806054, 0.1998824535359751, 0.27263065476519033, 5.470204806078328, 2.3764742830051633, 1.2476799856724419],
    [1.4676691264702488, 1.886335637072707, 0.02031542052264035, 2.581635451524263, 0.2778433459662767, 5.426157234141189, 2.7863155755092466, 0.47377331452948535],
    [4.731406825135263, 2.7877675239272075, 4.702453013483652, 3.627080115724877, 4.522233036199511, 9.76549473142389, 13.135203841406177, 4.811414303958371],
    [4.485616105900042, 4.713978903503316, 4.5142803069904005, 11.603401443903032, 11.136399702769415, 6.874793495833543, 6.741931433350623, 9.53429139110998],
    [3.5238826720577276, 3.1725512930351956, 1.4618735562214153, 0.7108629769818866, 0.7844653054311939, np.nan, 4.716221906356266, 2.705332173796982],
])


print(data.shape)

# Define base and target distribution names
#2d
#base_distributions = ["Diagonal Gaussian", "Uniform", "Uniform-Gaussian", "GMM", "Beta", "Gamma", "Exponential", "Poisson", "Student's T", "Cauchy", "Sawtooth"]
#target_distributions = ["Two Modes", "Two Moons", "Circular GMM", "Two Ring", "Spiral"]

#3d
#base_distributions = ["Diagonal Gaussian", "Uniform-Gaussian", "GMM", "Beta", "Gamma", "Exponential", "Poisson", "Student's T", "Cauchy", "Sawtooth"]
#target_distributions = ["Swiss Roll", "Klein Bottle", "Mobius Strip"]

#All
base_distributions = ["Diagonal Gaussian", "Uniform", "Uniform-Gaussian", "GMM", "Beta", "Gamma", "Exponential", "Poisson", "Student's T", "Cauchy", "Sawtooth"]
target_distributions = ["Two Modes", "Two Moons", "Circular GMM", "Two Ring", "Spiral", "Swiss Roll", "Klein Bottle", "Mobius Strip"]

# Create the heatmap
plt.figure(figsize=(15, 15))

cmap = plt.cm.cool
cmap.set_bad(color='gray') 

plt.imshow(data, cmap=cmap, interpolation='nearest')

# Add color bar
cbar = plt.colorbar()
cbar.set_label('Loss (KL-Divergence)', rotation=270, labelpad=15)

# Add titles and labels
plt.title('KL-Divergence Heatmap')
plt.xlabel('Target Distributions')
plt.ylabel('Base Distributions')

# Label the axes
plt.xticks(np.arange(len(target_distributions)), target_distributions)
plt.yticks(np.arange(len(base_distributions)), base_distributions)


for i in range(len(base_distributions)):
    for j in range(len(target_distributions)):
        plt.text(j, i, f'{data[i, j]:.2f}', 
                 ha='center', va='center', color='black')

# Save the plot as an image file
plt.savefig('kl_divergence_heatmap_all.png')

#display the plot if you want
# plt.show()

# If you don't need to display the plot, close it to free up memory
plt.close()
