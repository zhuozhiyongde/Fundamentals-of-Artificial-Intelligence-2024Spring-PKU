#pragma once

#include <vector>
#include <stack>
#include <unordered_map>

// 搜索树结点类
class SearchTreeNode{
private:
    
    // 为树中的结点建立唯一编号
    int _index;
    
    // 该结点有多少个子结点
    int _n_children;

    // 该结点的父结点
    SearchTreeNode* _parent;

    // 该结点的子结点列表
    std::vector<SearchTreeNode*> _children;

public:

    SearchTreeNode() = default;

    // 构造一个编号为index的结点，该结点没有父节点和子结点
    SearchTreeNode(int index) : _index(index), _parent(nullptr), _n_children(0) {}

    // 将child添加为当前结点的子结点
    void add_child(SearchTreeNode* child){
        ++ _n_children;
        _children.push_back(child);
        child->_parent = this;
    }

    // 返回结点编号
    int index() const {return _index;}

    // 当前结点有多少子结点
    int n_children() const {return _n_children;}

    // 当前结点的父节点
    SearchTreeNode* parent() const {return _parent;}

    // 当前结点的第child_index个子结点
    SearchTreeNode* child(int child_index) const {return _children[child_index];}

    // 当前结点的子节点列表
    const std::vector<SearchTreeNode*>& children() const {return _children;}

    // 比较结点相同只需要比较编号
    friend bool operator== (const SearchTreeNode& n1, const SearchTreeNode& n2){
        return n1._index == n2._index;
    }

    friend struct std::hash<SearchTreeNode>;
};

template<>
struct std::hash<SearchTreeNode>{
    size_t operator() (const SearchTreeNode& n) const {
        return n._index;
    }
};

// 搜索树类
class SearchTree{
private:
    
    // 单调递增的结点标识符
    int _unique_identifier;

    // 树当前的大小
    int _n_nodes;

    // 搜索树的根
    SearchTreeNode* _root;

    // 将结点的编号映射到结点指针
    std::unordered_map<int, SearchTreeNode*> _node_of;
    
public:

    // 构造一棵搜索树，树根为root，仅包含1个根结点
    SearchTree() : _root(new SearchTreeNode(0)), _unique_identifier(1), _n_nodes(1) {
        _node_of[0] = _root;
    }

    // 返回结点编号对应的结点指针
    SearchTreeNode* node_of(int index) const {
        return _node_of.at(index);
    }

    // 删除掉某棵子树之外的部分，更新树根和大小信息
    void destroy_except_subtree(SearchTreeNode* subtree_root){
        SearchTreeNode* node;
        std::stack<std::pair<SearchTreeNode*, int> > node_stack;
        node_stack.push(std::make_pair(_root, 0));
        std::pair<SearchTreeNode*, int> node_searched;
        int searched, deleted = 0;

        // 深度优先删除结点
        while (not node_stack.empty()){
            node_searched = node_stack.top();
            node_stack.pop();
            node = node_searched.first;
            searched = node_searched.second;

            // 如果当前探索到了不应当删除的子树的根，则不进行删除操作
            if (node == subtree_root){
                continue;
            }

            // 如果所有子结点已经删除，则删除该结点自身
            if (searched >= node->n_children()){
                ++ deleted;
                _node_of.erase(node->index());
                delete node;
            
            } else {
                node_stack.push(std::make_pair(node, searched+1));
                node_stack.push(std::make_pair(node->child(searched), 0));
            }
        }

        _n_nodes -= deleted;
        _root = subtree_root;
    }

    // 返回树根
    SearchTreeNode* root() const {return _root;}

    // 生成一个新的结点，这个结点和搜索树中已有的结点都不相同
    SearchTreeNode* create_node(){
        auto new_node = new SearchTreeNode(_unique_identifier);
        _node_of[_unique_identifier] = new_node;
        ++ _unique_identifier;
        return new_node;
    }

    // 将create_node生成的结点添加为parent的子结点，更新树的大小
    void add_as_child(SearchTreeNode* parent, SearchTreeNode* child){
        parent->add_child(child);
        ++ _n_nodes;
    }

    // 返回树当前的大小
    int n_nodes() const {return _n_nodes;}

    // 销毁整棵树（应当保证每个节点都是调用create_node生成的）
    ~SearchTree() {
        destroy_except_subtree(nullptr);
    }
};