#include <iostream>

#include <cuj/cuj.h>

using namespace cuj;

struct Node
{
    Node   *left  = nullptr;
    Node   *right = nullptr;
    int32_t key   = -1;
    int32_t value = -1;
};

CUJ_CLASS(Node, left, right, key, value) { using CUJBase::CUJBase; };

Pointer<Node> find_node(const Node *node, i32 value)
{
    if(!node)
        return ptr_literal<Node>(nullptr);

    Pointer<Node> result;
    $if(node->key == value)
    {
        result = ptr_literal(node);
    }
    $elif(node->key < value)
    {
        result = find_node(node->right, value);
    }
    $else
    {
        result = find_node(node->left, value);
    };
    return result;
}

void run()
{
    // build binary tree

    /*
                a
            b       c
        d   e       f   g
    */

    Node a, b, c, d, e, f, g;

    d.key = 0;
    b.key = 1;
    e.key = 2;
    a.key = 3;
    f.key = 4;
    c.key = 5;
    g.key = 6;

    a.value = a.key + 1;
    b.value = b.key + 1;
    c.value = c.key + 1;
    d.value = d.key + 1;
    e.value = e.key + 1;
    f.value = f.key + 1;
    g.value = g.key + 1;

    a.left  = &b;
    a.right = &c;

    b.left  = &d;
    b.right = &e;

    c.left  = &f;
    c.right = &g;

    // construct searching function

    ScopedContext ctx;

    to_callable<Pointer<Node>>("find", [&](i32 value)
    {
        $return(find_node(&a, value));
    });

    std::cout << ctx.gen_llvm_string(gen::LLVMIRGenerator::Target::Host)
              << std::endl;

    auto jit = ctx.gen_native_jit();
    auto find = jit.get_function_by_name<const Node *(int32_t)>("find");

    // test

    std::cout << find(a.key)->value << " == " << a.value << std::endl;
    std::cout << find(c.key)->value << " == " << c.value << std::endl;
    std::cout << find(e.key)->value << " == " << e.value << std::endl;
    std::cout << find(100)          << " == " << nullptr << std::endl;
}

int main()
{
    try
    {
        run();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}
