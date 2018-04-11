from collections import defaultdict

def eval_rouge(system_sens, ref_sens, max_words=100):
  ref_dist = defaultdict(int)
  ref_words = 0
  for sen in ref_sens:
    for word in sen:
      ref_dist[word] += 1
      ref_words += 1
      if ref_words > max_words: break
    if ref_words > max_words: break
  sys_dist = defaultdict(int)
  sys_words = 0
  for sen in system_sens:
    for word in sen:
      sys_dist[word] += 1
      sys_words += 1
      if sys_words > max_words: break
    if sys_words > max_words: break
  covered = 0
  for key in ref_dist:
    covered += min(sys_dist[key],ref_dist[key])
  return float(covered) / ref_words
