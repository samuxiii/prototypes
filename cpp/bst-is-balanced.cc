#include <iostream>
#include <cmath>

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
    std::cout << "inorder: ";
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

bool isBalanced()
{
    if (isHeightBalanced(root) == -1)
        return false;
    return true;
}

int isHeightBalanced(Node* n)
{
    if (!n) return 0;
    
    int hl = isHeightBalanced(n->left);
    int hr = isHeightBalanced(n->right);
    
    if (hl == -1 || hr == -1) return -1;
    if (std::abs(hl - hr) > 1) return -1;
    if (hl > hr) return hl+1;
    else return hr+1;
}
};

int main()
{
    BST bst;
    bst.insert(3);
    bst.insert(1);
    bst.insert(4);
    bst.print();
    std::cout << "Balanced: " << (bst.isBalanced()?"yes":"no") << std::endl;
    
    BST bst2;
    bst2.insert(3);
    bst2.insert(1);
    bst2.insert(2);
    bst2.print();
    std::cout << "Balanced: " << (bst2.isBalanced()?"yes":"no") << std::endl;

}
