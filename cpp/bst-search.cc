#include <iostream>
#include <queue>
#include <stack>

struct BST
{
struct Node
{
    int key;
    Node* left;
    Node* right;
    
    Node(int k):key(k){};
};

Node* root = nullptr;

void insert(int key)
{
    root = insert(root, key);        
}

Node* insert(Node* n, int key)
{
    if(!n) 
       return new Node(key);
    
    if(key < n->key)
        n->left = insert(n->left, key);
    else if (key > n->key)
        n->right = insert(n->right, key);
    else
        n->key = key;
        
    return n;
}

void print()
{
    std::cout << "inorder:"<< std::endl;
    inorder(root);
    std::cout << std::endl;
}
void inorder(Node* n)
{
    if (!n) return;
    inorder(n->left);
    std::cout << n->key << " ";
    inorder(n->right);
}
};


//we use container stack
void depthFirstSearch(BST bst)
{
    std::stack<BST::Node*> s;
    //store root in container
    s.push(bst.root);
    
    std::cout << "Depth First:" << std::endl;
    
    while(!s.empty())
    {
        auto node = s.top();
        s.pop();
                
        if (node != nullptr)
        {        
            //store children
            s.push(node->right);
            s.push(node->left);
            
            //do something with node
            std::cout << node->key << std::endl;
        }
    }
}

//we use container queue
void breadthFirstSearch(BST bst)
{
    std::queue<BST::Node*> q;
    //store root in container
    q.push(bst.root);
    
    std::cout << "Breadth First:" << std::endl;
    
    while(!q.empty())
    {
        auto node = q.front();
        q.pop();
                
        if (node != nullptr)
        {        
            //store children
            q.push(node->left);
            q.push(node->right);
            
            //do something with node
            std::cout << node->key << std::endl;
        }
    }
}

int main()
{
    BST bst;
    bst.insert(5);
    bst.insert(3);
    bst.insert(2);
    bst.insert(4);
    bst.insert(7);
    bst.insert(6);
    bst.insert(8);
    bst.print();
 
    depthFirstSearch(bst);
    breadthFirstSearch(bst);
}