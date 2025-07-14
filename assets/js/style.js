/*!
 * Start Bootstrap – Resume v4.0.0-beta.2
 * https://startbootstrap.com/template-overviews/resume
 * Copyright 2013-2017 Start Bootstrap
 * Licensed under MIT
 * https://github.com/BlackrockDigital/startbootstrap-resume/blob/master/LICENSE
 */

(function ($) {
  "use strict";

  // ------------------------------------------------------------
  // Smooth scrolling with jQuery Easing for links that have the
  // class .js-scroll-trigger and point to an in-page anchor.
  // ------------------------------------------------------------
  $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function () {
    // Confirm that the clicked link is on the same page
    if (
      location.pathname.replace(/^\//, "") === this.pathname.replace(/^\//, "") &&
      location.hostname === this.hostname
    ) {
      // Identify the target element
      var $target = $(this.hash);
      $target = $target.length ? $target : $("[name=" + this.hash.slice(1) + "]");

      if ($target.length) {
        // Animate the scroll
        $("html, body").animate(
          { scrollTop: $target.offset().top },
          1000,
          "easeInOutExpo"
        );
        return false; // Prevent default link behaviour
      }
    }
  });

  // ------------------------------------------------------------
  // Close the responsive menu when any .js-scroll-trigger link
  // is clicked (useful for collapsed mobile navbars).
  // ------------------------------------------------------------
  $(".js-scroll-trigger").click(function () {
    $(".navbar-collapse").collapse("hide");
  });

  // ------------------------------------------------------------
  // Activate Bootstrap Scrollspy on the body element to add
  // an active class to navbar items as the page is scrolled.
  // ------------------------------------------------------------
  $("body").scrollspy({
    target: "#sideNav"
  });

})(jQuery);