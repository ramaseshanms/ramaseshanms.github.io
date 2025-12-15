// blogs-list.js
// Order: newest first
const blogs = [
  {
    title: "FMA-Net++",
    file: "blogs/FMA-Net++.md",
    date: "2025-12-16"
  },
  {
    title: "Blog 2 Title",
    file: "blogs/blog2.md",
    date: "2025-11-01"
  },
  {
    title: "Blog 1 Title",
    file: "blogs/blog1.md",
    date: "2025-09-15"
  }
];

// Sort blogs by date descending (newest first)
blogs.sort((a, b) => new Date(b.date) - new Date(a.date));
