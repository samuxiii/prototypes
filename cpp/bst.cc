// Example program
#include <iostream>

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

Node* search(int key)
{
    return search(root, key);
}

Node* search(Node* n, int key)
{
    if (!n) return nullptr;
    
    if (key < n->key)
        return search(n->left, key);
    else if (key > n->key)
        return search(n->right, key);
    else
        return n;
}

Node* minValueNode (Node* n)
{
    Node* current = n;
    
    while(current->left)
        current = current->left;
        
    return current;
}

void del(int key)
{
    root = del(root, key);
}

Node* del(Node* n, int key)
{
    if (!n) return nullptr;
    
    if (key < n->key)
        n->left = del(n->left, key);
    else if (key > n->key)
        n->right = del(n->right, key);
    else
    {
        if (!n->left)
        {
            Node* tmp = n->right;
            delete n;
            return tmp;
        }
        else if (!n->right)
        {
            Node* tmp = n->left;
            delete n;
            return tmp;
        }
        
        //two childs
        Node* tmp = minValueNode(n->right);
        
        n->key = tmp->key;
        
        n->right = del(n->right, tmp->key);
    }
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
    
    auto n = bst.search(2);
    std::cout << "key: " << n->key << std::endl;
    
    bst.del(2);
    bst.print();
    bst.del(3);
    bst.print();
    bst.del(5);
    bst.print();

}

