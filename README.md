### implicit
---
https://github.com/benfred/implicit


```py
// tests/recommender_base_test.py
from __future__ import print_function

import pickle

import numpy as np
from scipy.sparse import csr_matrix

from implicit.evaluation import precision_at_k
from implicit.nearest_neighbours import ItemItemRecommender

class TestRecommenderBaseMixin(object):
  """ """
  def _get_model(self):
    raise NotImplementedError()
    
  def test_recommend(self):
    item_users = self.get_checker_board(50)
    user_items = item_users.T.tocsr()
    
    model = self._get_model()
    model.fit(item_users, show_progress=False)
    
    for userid in range(50):
      recs = model.recommend(userid, user_items, N=1)
      self.assertEqual(len(recs), 1)
      
      self.assertEqual(recs[0][0], userid)
      
    recs = model.recommend(0, user_items, N=10000)
    self.assertTrue(len(recs))
    
  def test_recalculate_user(self):
    item_users = self.get_checker_board(50)
    user_items = item_users.T.tocsr()
    
    model = self._get_model()
    model.fit(item_user, show_progress=False)
    
    for userid in range(item_users.shape[1]):
      recs = model.recommend(userid, user_items, N=1)
      self.assertEqual(len(recs), 1)
      user_vector = user_items[userid]
      
      try:
        recs_from_liked = model.recommend(userid=0, user_items=user_vector,
          N=1, recalculate_user=True)
        self.assertEqual(recs[0][0], recs_from_liked[0][0])
        
        self.assertAlmostEqual(recs[0][1], recs_from_liked[0][1], places=4)
      except NotImplementedError:
        pass
        
  def test_evaluation(self):
    item_users = self.get_checker_board(50)
    user_items = item_users.T.tocsr()
    
    model = self._get_model()
    model.fit(item_users, show_progress=False)
    
    p = precision_at_k(model, user_items.tocsr(), csr_matric(np.eye(50)), K=1,
      show_progress=False)
    self.assertEqual(p, 1)
    
  def test_similar_users(self):
    
    model = self._get_model()
    if isinstance(model, ItemItemRecommender):
    
      


```

```
```

```
```

