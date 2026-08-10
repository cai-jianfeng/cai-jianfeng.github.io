// Add one-click copy buttons to BibTeX blocks on publication pages.
(function($) {
  "use strict";

  $(function() {
    if (window.location.pathname.indexOf("/publication/") !== 0) {
      return;
    }

    function legacyCopy(text) {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "absolute";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      var ok = false;
      try {
        ok = document.execCommand("copy");
      } catch (err) {
        ok = false;
      }
      document.body.removeChild(textarea);
      return ok;
    }

    function copyText(text, done) {
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function() {
          done(true);
        }, function() {
          // writeText can reject when the document is unfocused or the
          // permission is denied; fall back to the legacy path.
          done(legacyCopy(text));
        });
        return;
      }
      done(legacyCopy(text));
    }

    $(".page__content pre").each(function() {
      var $pre = $(this);
      // Capture the citation before injecting the button so the copied
      // text never contains the button label.
      var bibtex = $pre.text().replace(/^\s+|\s+$/g, "");
      if (bibtex.charAt(0) !== "@") {
        return;
      }

      $pre.addClass("bibtex-pre");
      var $button = $("<button>", {
        type: "button",
        "class": "bibtex-copy-btn",
        title: "Copy BibTeX",
        "aria-label": "Copy BibTeX citation to clipboard",
        html: '<i class="fa fa-clipboard" aria-hidden="true"></i><span class="screen-reader-text">Copy BibTeX</span>'
      });
      $button.on("click", function() {
        copyText(bibtex, function(ok) {
          $button
            .toggleClass("is-copied", ok)
            .attr("title", ok ? "Copied!" : "Copy failed")
            .find(".fa").attr("class", ok ? "fa fa-check" : "fa fa-times");
          window.setTimeout(function() {
            $button
              .removeClass("is-copied")
              .attr("title", "Copy BibTeX")
              .find(".fa").attr("class", "fa fa-clipboard");
          }, 2000);
        });
      });
      $pre.append($button);
    });
  });
})(jQuery);
