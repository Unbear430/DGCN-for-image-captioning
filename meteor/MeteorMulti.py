import numpy as np


def producer_fn(q, scorer, gts, res, vid_order):
  _, ss = scorer.compute_score(gts, res, vid_order=vid_order)
  vid_ss = {}
  for vid, s in zip(vid_order, ss):
    vid_ss[vid] = s
  q.put(vid_ss)

class MeteorMulti(object):
  def __init__(self, num_process=4):
    self.num_process = num_process
    self.scorers = []
    for i in xrange(num_process):
      self.scorers.append(Meteor())

  def compute_score(self, gts, res, vid_order=None):
    if vid_order is None:
      vid_order = gts.keys()
    num_vid = len(vid_order)
    num_split = min(self.num_process, num_vid)
    split_idxs = np.linspace(0, num_vid, num_split+1).astype(np.int32)

    q = Queue(num_split)
    producers = []
    for i in xrange(num_split):
      sub_vid_order = vid_order[split_idxs[i]: split_idxs[i+1]]
      sub_gts = {key: gts[key] for key in sub_vid_order}
      sub_res = {key: res[key] for key in sub_vid_order}

      producers.append(Process(target=producer_fn,
        args=(q, self.scorers[i], sub_gts, sub_res, sub_vid_order)))
      producers[-1].start()

    vid_score = {}
    for i in xrange(num_split):
      sub_vid_ss = q.get()
      vid_score.update(sub_vid_ss)
    scores = [vid_score[vid] for vid in vid_order]

    return np.mean(scores), scores