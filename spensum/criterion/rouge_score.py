from collections import defaultdict

def eval_rouge(system_sens, ref_sens, max_words=100, stopwords=set()):

  if type(system_sens) is defaultdict and type(red_sens) is defaultdict:
    sys_dist = system_sens
    ref_dist = ref_sens
  else: 
    ref_dist = defaultdict(int)
    ref_words = 0
    for sen in ref_sens:
      for word in sen:
        if word.lower() not in stopwords:
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
  score = float(covered) / ref_words if ref_words >= 10 else 0
  #print("DEBUG: rouge eval score %f" % score)
  return (score,sys_dist,ref_dist)
  
