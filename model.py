import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_actions = num_actions
        num_dim = len(state_low)
        self.num_dim = num_dim

        self.tiling_dims = np.zeros(num_dim) # number of tiles in each dim
        for dim in range(num_dim):
            nt = (state_high[dim] - state_low[dim]) / tile_width[dim]
            self.tiling_dims[dim] = np.ceil(np.around(nt, decimals=2)) + 1

        ts = [num_actions, num_tilings]
        for d in self.tiling_dims:
            ts.append(int(d))
        self.tilings = np.zeros((ts))
        
        self.tiling_start = np.zeros((num_tilings, num_dim))
        for t in range(num_tilings):
            for d in range(num_dim):
                self.tiling_start[t][d] = state_low[d] - ((t * tile_width[d]) / num_tilings)
                self.tiling_start[t][d] = np.around(self.tiling_start[t][d], decimals=5)

        self.f_vec_len = num_actions * num_tilings
        for dim in self.tiling_dims:
             self.f_vec_len *= dim
        self.f_vec_len = int(self.f_vec_len)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.f_vec_len


    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d

        Returns which features are activated for a given state and action
        """
        x = np.zeros(self.tilings.shape)
        if done:
            return x.flatten()
        
        curr_tiling = x[a] # only for that action

        for i_t, tiling in enumerate(curr_tiling):
            d0 = int(np.floor( (s[0] - self.tiling_start[i_t][0])/self.tile_width[0] ))
            d1 = int(np.floor( (s[1] - self.tiling_start[i_t][1])/self.tile_width[1] ))
            x[a][i_t][d0][d1] = 1
        
        return x.flatten()
