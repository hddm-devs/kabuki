import kabuki
import pymc as pm
import numpy as np

data = kabuki.utils.generate_effect_data(2, 1, .5, -1.5, 15, 50)

# Generate model
model = kabuki.ANOVA(data, is_subj_model=True, depends_on={'effect':['cond']})
model.mcmc(map_=False)
print model.summary()
