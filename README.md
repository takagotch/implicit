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
      return
    model.fit(self.get_checker_board(50), show_progress=False)
    for userid in range(50):
      recs = model.similar_users(userid, N=10)
      for r, _ in recs:
        self.assertEqual(r % 2, userid % 2)
        
  def test_similar_items(self):
    model = self._get_model()
    model.fit(self.get_checker_board(50), show_progress=False)
    for itemid in range(50):
      recs = model.similar_items(itemid, N=10)
      for r, _ in recs:
        self.assertEqual(r % 2, itemid % 2)
        
  def test_zero_length_row(self):
    item_users = self.get_checker_board(50).todense()
    item_users[42] = 0
    item_users[:, 42] = 0
    
    item_users[49] = 0
    item_users[:, 49] = 0
    
    model = self._get_model()
    model.fit(csr_matrix(item_users), show_progress=False)
    
    for itemid in range(40):
      recs = model.similar_items(itemid, 10)
      self.assertTrue(42 not in [r for r, _ in recs])
      
  def test_dtype(self):
    item_users = self.get_checker_board(50)
    model = self._get_model()
    model.fit(item_users.astype(np.float64), show_progress=False)
    
    model = self._get_model()
    model.fit(item_users.astype(np.float32), show_progress=False)
    
  def test_rank_items(self):
    item_users = self.get_checker_board(50)
    user_items = item_users.T.tocsr()
    
    model = self._get_model()
    model.fit(item_users, show_progress=False)
    
    for userid in range(50):
      selected_items = np.random.randint(50, size=10).tolist()
      ranked_list = model.rank_items(userid, user_items, selected_items)
      ordered_items = [itemid for (itemid, score) in ranked_list]
      
      self.asertEqual(set(ordered_items), set(selected_items))
      
      wrong_neg_items = [-1, -3, -5]
      wrong_pos_items = [51, 300, 200]
      
      with self.assertRaises(IndexError):
        wrong_item_list = selected_items + wrong_neg_items
        model.rank_items(userid, user_items, wrong_item_list)
      with self.assertRaises(IndexError):
        wrong_item_list = selected_items + wrong_pos_items
        model.rank_items(userid, user_items, wrong_item_list)
        
  def test_pickle(self):
    item_users = self.get_checker_board(50)
    model = self._get_model()
    model.fit(item_users, show_progress=False)
    
    pickled = pickle.dumps(model)
    pickle.loads(pickled)
    
  def get_checker_board(self, X):
    """
    """
    for in in range(X):
      for j in range(i % 2, X, 2):
        ret[i, j] = 1.0
    return csr_matrix(ret - np.eye(X))
```

```
```

```
```

