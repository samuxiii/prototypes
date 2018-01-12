// Example program
#include <iostream>
#include <string>
#include <random>

struct Node 
{
    int data;
    Node* next = nullptr;
};

struct Queue
{
    Node* head = nullptr;
    Node* tail = nullptr;
    
    void enqueue(int x)
    {
        if (!head)
        {
            head = new Node();
            head->data = x;
            tail = head;
        }
        else
        {
            tail->next = new Node();
            tail->next->data = x;
            tail = tail->next;
        }
    }
    
    int dequeue()
    {
        if (!head)
        {
            return -1;
        }
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
    Queue q;
    
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    q.enqueue(4);
    q.print();
    q.dequeue();
    q.dequeue();
    q.print();
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    q.enqueue(4);
    q.print();
    q.dequeue();
    q.print();

}

