const blogsDiv = document.getElementById("blogs");

function loadBlogs() {
    blogsDiv.innerHTML = "";
    const blogs = JSON.parse(localStorage.getItem("blogs")) || [];

    blogs.reverse().forEach((blog, index) => {
        const div = document.createElement("div");
        div.className = "card";

        div.innerHTML = `
            <h3>${blog.title}</h3>
            <span>${blog.date}</span>
            <p style="white-space:pre-line;margin-top:10px">${blog.content}</p>
            <button onclick="deleteBlog(${index})" style="margin-top:10px">
                Delete
            </button>
        `;

        blogsDiv.appendChild(div);
    });
}

function addBlog() {
    const title = document.getElementById("title").value.trim();
    const content = document.getElementById("content").value.trim();

    if (!title || !content) {
        alert("Please add title and content");
        return;
    }

    const blogs = JSON.parse(localStorage.getItem("blogs")) || [];

    blogs.push({
        title,
        content,
        date: new Date().toLocaleDateString()
    });

    localStorage.setItem("blogs", JSON.stringify(blogs));

    document.getElementById("title").value = "";
    document.getElementById("content").value = "";

    loadBlogs();
}

function deleteBlog(index) {
    const blogs = JSON.parse(localStorage.getItem("blogs")) || [];
    blogs.splice(blogs.length - 1 - index, 1);
    localStorage.setItem("blogs", JSON.stringify(blogs));
    loadBlogs();
}

loadBlogs();
