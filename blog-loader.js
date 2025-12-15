async function loadBlog(mdFile) {
    const response = await fetch(mdFile);
    const text = await response.text();
    document.getElementById("blog-content").innerHTML = marked.parse(text);
}

// Example: Load your first blog
loadBlog("blogs/FMA-Net++.md");
