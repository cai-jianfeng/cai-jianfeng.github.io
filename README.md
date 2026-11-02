# Jianfeng Cai's Academic Homepage

Personal academic website of **Jianfeng Cai (蔡建峰)**, an M.S. candidate in Information and Communication Engineering at the University of Science and Technology of China (USTC). His research focuses on post-training of large language models, reinforcement learning, and coding agents. The site collects research, publications, internship experience, education, and technical notes.

**Live site:** <https://cai-jianfeng.github.io>

## Site Sections

- **Home** — introduction, selected publications and internships, education, technical skills, and featured blog posts
- **Publications** — selected papers and technical reports
- **Internships** — research and engineering experience in LLM post-training, agents, retrieval-augmented generation, and 3D vision
- **Education** — graduate and undergraduate education
- **Blog Posts** — technical notes, paper reviews, and hands-on explorations in machine learning and systems

## Technology

- [Jekyll](https://jekyllrb.com/) built against the `github-pages` gem, using the [academicpages](https://academicpages.github.io/) template, which is derived from the [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) theme
- Kramdown for Markdown, Sass for styling, and a small amount of JavaScript for navigation and interactive elements
- GitHub Pages for hosting and automatic builds
- Jupyter notebooks and Python scripts in `markdown_generator/` for producing content from TSV files

## Local Development

Requirements: Ruby 3.x with Bundler. Node.js is optional and only needed to rebuild the minified JavaScript bundle.

```bash
bundle install
bundle exec jekyll serve --livereload --config "_config.yml,_config.dev.yml"
```

Then open <http://localhost:4000>. `_config.dev.yml` points the site at localhost and disables analytics. On Windows, `serve-local.cmd` is a convenience wrapper around `serve-local.ps1` that runs the same command with file-watching polling enabled.

## Repository Layout

| Content | Location |
| --- | --- |
| Site settings, author profile, and collections | `_config.yml` |
| Navigation menu | `_data/navigation.yml` |
| Blog posts | `_posts/YYYY-MM-DD-title.md` |
| Publications | `_publications/` |
| Internships | `_internships/` |
| Education | `_teaching/` |
| Standalone pages | `_pages/` |
| Images and downloadable files | `images/`, `files/` |
| Layouts, includes, and styles | `_layouts/`, `_includes/`, `_sass/`, `assets/` |
| Bulk content generation | `markdown_generator/` |

Content files use YAML front matter with fields such as `title`, `layout`, `permalink`, and `date`. Collections are configured under `collections` in `_config.yml`; add a new Markdown file with the same front matter to add an entry.

## Deployment

The site is published with GitHub Pages from the `master` branch. Merge changes into `master` and GitHub Pages builds and deploys the site automatically; no manual build step is required.

## Credits and License

This site is based on the [academicpages](https://academicpages.github.io/) template by Stuart Geiger, which was forked from the [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) Jekyll theme by Michael Rose. Both are released under the MIT License; see [LICENSE](LICENSE). The upstream theme changelog and contribution notes are preserved in [CHANGELOG.md](CHANGELOG.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

- Email: [xiaobaicai@mail.ustc.edu.cn](mailto:xiaobaicai@mail.ustc.edu.cn) or [cjf1622613693@gmail.com](mailto:cjf1622613693@gmail.com)
- QQ: 1622613693
- WeChat: 19959509839
