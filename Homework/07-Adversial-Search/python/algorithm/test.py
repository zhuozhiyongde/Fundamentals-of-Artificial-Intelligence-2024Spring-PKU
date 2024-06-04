def _sample_path(self, state: GameStateBase, exploration: float) -> np.ndarray:
    index = self._state_to_index[state]
    node = self._tree.node_of[index]
    self._visit_count_of[index] += 1

    # 还可以扩展
    if node.n_children < state.n_actions():
        next_state = state.next(state.action_space()[node.n_children])
        child = self._tree.create_node()
        self._tree.add_as_child(node, child)
        self._state_to_index[next_state] = child.index
        self._index_to_state[child.index] = next_state
        self._visit_count_of[child.index] = 1
        # TODO：子结点初始累计收益 values 为模拟得到的值——1行
        values = self._simulate_from(next_state)
        self._value_sums_of[child.index] = values
    elif node.n_children > 0:
        selection = MaxSelection()
        selection.initialize(node.n_children, -float("inf"))
        for i in range(node.n_children):
            child = node.child(i).index
            # TODO：选择UCT值最大的子结点继续探索
            selection.submit(
                self._value_sums_of[child][state.active_player()]
                / self._visit_count_of[child]
                + exploration
                * sqrt(
                    log(self._visit_count_of[index]) / self._visit_count_of[child]
                )
            )
        next_state = state.next(state.action_space()[selection.selected_index()])
        values = self._sample_path(next_state, exploration)
    else:
        values = np.array(state.cumulative_rewards(), dtype=np.float64)

    self._value_sums_of[index] += values
    return values

def select_action(self, iterations: int, exploration: float):
    root_state = self._index_to_state[0]
    for i in range(iterations):
        self._sample_path(root_state, exploration)

    root = self._tree.root
    selection = MaxSelection()
    selection.initialize(root.n_children, -float("inf"))

    for i in range(root.n_children):
        child = root.child(i).index
        # TODO：按平均价值贪心选择
        selection.submit(
            self._value_sums_of[child][root_state.active_player()]
            / self._visit_count_of[child]
        )

    return root_state.action_space()[selection.selected_index()]