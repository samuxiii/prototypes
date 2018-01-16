#include <iostream>
#include <string>

struct Node 
{
    int data;
    Node* next = nullptr;
};

struct Stack
{
    Node* head = nullptr;
    Node* tail = nullptr;
    
    void push (int x)
    {
        if (!head)
        {
            head = new Node();
            head->data = x;
            tail = head;
        }
        else
        {
            Node* tmp = new Node();
            tmp->data = x;
            tmp->next = head;
            head = tmp;
        }
    }
    
    int pop()
    {
        if (!head)
            return -1;
        else
        {
            int tmp = head->data;
            Node* old = head;
            head = head->next;
            delete old;
            return tmp;
        }
    }
    
    void print()
    {
        std::cout << "===" << std::endl;
        Node* temp = head;
        while (temp != nullptr)
        {
            std::cout << temp->data << std::endl;
            temp = temp->next;
        }
    }
};

int main()
{
    Stack s;
    
    s.push(1);
    s.push(2);
    s.push(3);
    s.push(4);
    s.print();
    s.pop();
    s.pop();
    s.print();
    s.push(1);
    s.push(2);
    s.push(3);
    s.push(4);
    s.print();
    s.pop();
    s.print();

}

