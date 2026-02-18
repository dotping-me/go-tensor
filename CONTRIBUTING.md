# Contributing to Go CPU-First Deep Learning Framework
Thank you for your interest in contributing! 🎉

This project aims to build a transparent CPU-first deep learning framework in pure Go **for now!** *(C optimizations will be added afterwards but not now)*. Contibutions are welcome as long as they progress the project forward, even if it means writing proper docs 😅😂

## 🛠 How to Contribute
Fork the repository, run the project locally and submit PRs!

### 1. Submitting Contributions
1. **Fork the Repository**
    - There's a fork button on the GitHub page!

2. **Clone your Fork**
    ```bash
    git clone https://github.com/YOUR_USERNAME/go-tensor.git
    ```

3. **CD into the Project**
    ```bash
    cd go-tensor/
    code . # I'm using VSCode here
    ```

4. **Install Dependencies**
    - There are none and let's keep it that way; minimal! But still it's good practice so:
        ```bash
        go mod tidy
        ```
5. **Commit, Push & Submit a PR**
    - Commit and Push your changes to *your forked repository*.
    - Open a Pull Request:
        - Head to the GitHub page of your forked repository.
        - Click on `Compare Changes`.
        - Click on `Open a pull request` to submit your changes!

### 2. Reporting Issues and Suggestions
Identifying bugs and areas for improvements would be extremely helpful, so please open an issue on GitHub and provide as much details as you can; logs, screenshots, etc...

## 📏 Guidelines
1. Review, add and run test cases for your contributions!
2. Document as much as you can the concepts you are implementing, including the mathematics behind it!
3. Use clear and concise commit messages.
    - Follow the format `type(scope): description`.
        - i.e. `fix(embedding): Fixed shape misalignment issue in embedding layer`.
        - Use types such as `feat`, `fix`, `docs`, `refactor` and `test`.
4. Review that your code adheres to guidelines before submitting a PR!
5. Keep dependencies to a minimum, **ideally none!**

&nbsp;
---
### Thank you for being a part of this project! ✨