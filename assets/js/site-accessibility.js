(function($) {
  "use strict";

  $(function() {
    var $navButton = $("#site-nav button");
    var $hiddenLinks = $("#site-nav-hidden-links");
    var $authorButton = $(".author__urls-wrapper button");
    var $authorLinks = $("#author-profile-links");

    function syncNavigationState() {
      var expanded = !$hiddenLinks.hasClass("hidden");
      $navButton
        .toggleClass("close", expanded)
        .attr("aria-expanded", expanded ? "true" : "false");
    }

    function syncAuthorState() {
      var expanded = $authorButton.is(":visible") &&
        $authorButton.hasClass("open") &&
        $authorLinks.is(":visible");

      if (!expanded) {
        $authorButton.removeClass("open");
      }

      $authorButton.attr("aria-expanded", expanded ? "true" : "false");
    }

    $navButton.on("click.siteAccessibility", syncNavigationState);
    $authorButton.on("click.siteAccessibility", syncAuthorState);

    $(window).on("resize.siteAccessibility", function() {
      window.setTimeout(function() {
        syncNavigationState();
        syncAuthorState();
      }, 0);
    });

    $(document).on("keydown.siteAccessibility", function(event) {
      if (event.key === "Escape" && !$hiddenLinks.hasClass("hidden")) {
        $hiddenLinks.addClass("hidden");
        $navButton.removeClass("close").attr("aria-expanded", "false").focus();
      } else if (event.key === "Escape" && $authorButton.attr("aria-expanded") === "true") {
        $authorLinks.stop(true, true).hide();
        $authorButton.removeClass("open").attr("aria-expanded", "false").focus();
      }
    });

    syncNavigationState();
    syncAuthorState();
  });
})(jQuery);
