import pickle

def is_picklable(obj):
  try:
    pickle.dumps(obj)

  except pickle.PicklingError:
    return False
  return True

from parametric_tsne import ParametricTSNE
ptsne = ParametricTSNE()
print(is_picklable(ptsne))